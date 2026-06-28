"""FastAPI application for outpatient PET-CT visualization."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from petct.data import build_record_key
from petct.display_plan import (
    DisplayDetailField,
    DisplayFindingType,
    DisplayPlan,
    ReviewStrength,
    build_laterality_plan_from_display_plan,
    normalize_display_selection,
)
from petct.evaluation import HumanEvaluation
from petct.generation import (
    GenerationPipeline,
    GenerationRequest,
    PipelineStageError,
    ProgressCallback,
)
from petct.laterality import LateralityPlan, build_laterality_plan, merge_laterality_plans
from petct.model_catalog import fetch_openai_compatible_models
from petct.provider_config import BindingOverride, ProviderRegistry, ResolvedBinding
from petct.providers import (
    OpenAIDisplayPlanner,
    OpenAIImageGenerator,
    OpenAILateralityPlanner,
    OpenAIVisualReviewer,
)
from petct.provenance import build_reproducibility_metadata
from petct.question_service import OpenAIQuestionService
from prompts import PET_CT_IMG2IMG_PROMPT, PET_CT_VISUALIZATION_PROMPT


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

STATIC_DIR = BASE_DIR / "webapp" / "static"
RUNTIME_DIR = BASE_DIR / "runtime" / "cases"
DEFAULT_PROVIDER_CONFIG = BASE_DIR / "settings" / "providers.json"
LOCAL_PROVIDER_CONFIG = BASE_DIR / "settings" / "local_providers.json"
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PROVIDER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,80}$")
RUN_FILE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
INVALID_CASE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
REFERENCE_IMAGE_DATA_URL_PATTERN = re.compile(
    r"^data:(image/(?:png|jpe?g|webp));base64,(.+)$",
    re.IGNORECASE | re.DOTALL,
)
REFERENCE_IMAGE_MAX_BYTES = 15 * 1024 * 1024


def _compose_structured_report_text(conclusion_text: str, findings_text: str | None) -> str:
    parts = [f"检查结论：\n{conclusion_text.strip()}"]
    if findings_text and findings_text.strip():
        parts.append(f"检查所见：\n{findings_text.strip()}")
    return "\n\n".join(parts)


def _effective_review_strength(payload: "GeneratePayload") -> ReviewStrength:
    if payload.review_strength is not None:
        return ReviewStrength(payload.review_strength)
    return ReviewStrength.STANDARD if payload.gate_enabled else ReviewStrength.OFF


class GeneratePayload(BaseModel):
    case_id: str = Field(min_length=1, max_length=100)
    report_text: str | None = Field(default=None, max_length=30000)
    conclusion_text: str | None = Field(default=None, max_length=30000)
    findings_text: str | None = Field(default=None, max_length=30000)
    display_finding_types: list[DisplayFindingType] | None = None
    display_detail_fields: list[DisplayDetailField] | None = None
    gate_enabled: bool = True
    review_strength: ReviewStrength | None = None
    generate_questions: bool = True
    pipeline_id: str | None = None
    image_override: BindingOverride | None = None
    review_override: BindingOverride | None = None
    question_override: BindingOverride | None = None
    reference_image_data_url: str | None = Field(default=None, max_length=25_000_000)
    max_revisions: int | None = Field(default=None, ge=0, le=5)
    experiment_id: str | None = Field(default=None, max_length=100)
    experiment_item_id: str | None = Field(default=None, max_length=100)
    cancer_type: str | None = Field(default=None, max_length=100)
    random_seed: int | None = None
    laterality_mode: Literal["auto", "ai", "script"] = "auto"

    @model_validator(mode="after")
    def validate_report_inputs(self):
        if self.conclusion_text is not None:
            conclusion = self.conclusion_text.strip()
            findings = self.findings_text.strip() if self.findings_text else ""
            if not conclusion:
                raise ValueError("检查结论不能为空")
            self.conclusion_text = conclusion
            self.findings_text = findings or None
            self.report_text = (
                self.report_text.strip()
                if self.report_text and self.report_text.strip()
                else _compose_structured_report_text(conclusion, findings)
            )
        elif self.report_text is not None:
            self.report_text = self.report_text.strip()
        else:
            raise ValueError("report_text or conclusion_text is required")

        if len(self.report_text or "") < 10:
            raise ValueError("report_text must contain at least 10 characters")
        normalize_display_selection(
            self.display_finding_types,
            self.display_detail_fields,
        )
        return self


class ProviderPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=100)
    api_style: Literal["openai", "openai_compatible"] = Field(
        default="openai_compatible",
        alias="apiStyle",
    )
    api_key: str | None = Field(default=None, alias="apiKey")
    base_url: str | None = Field(default=None, alias="baseUrl")
    homepage: str | None = None
    notes: str | None = None
    timeout_seconds: float = Field(default=300, alias="timeoutSeconds", gt=0)
    cached_models: list[str] = Field(default_factory=list, alias="cachedModels")
    task_models: dict[str, str] = Field(default_factory=dict, alias="taskModels")


class FetchModelsPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider_id: str | None = Field(default=None, alias="providerId")
    api_key: str | None = Field(default=None, alias="apiKey")
    base_url: str | None = Field(default=None, alias="baseUrl")
    timeout_seconds: float | None = Field(default=None, alias="timeoutSeconds", gt=0)
    cache_provider_id: str | None = Field(default=None, alias="cacheProviderId")


def _provider_config_path() -> Path:
    configured = os.getenv("PETCT_PROVIDER_CONFIG", "").strip()
    return Path(configured) if configured else DEFAULT_PROVIDER_CONFIG


def _local_provider_config_path() -> Path:
    configured = os.getenv("PETCT_LOCAL_PROVIDER_CONFIG", "").strip()
    return Path(configured) if configured else LOCAL_PROVIDER_CONFIG


def _load_registry() -> ProviderRegistry:
    try:
        return ProviderRegistry.load(
            _provider_config_path(),
            local_path=_local_provider_config_path(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Provider configuration is invalid: {exc}",
        ) from exc


def _load_local_providers() -> dict:
    path = _local_provider_config_path()
    if not path.exists():
        return {"version": 1, "providers": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local provider configuration is invalid: {exc}",
        ) from exc
    data.setdefault("version", 1)
    data.setdefault("providers", [])
    return data


def _save_local_providers(data: dict) -> None:
    path = _local_provider_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _provider_id_from_label(label: str, existing_ids: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_.-]+", "-", label.lower()).strip("-")
    if not base:
        base = f"provider-{uuid.uuid4().hex[:8]}"
    provider_id = base[:80]
    suffix = 2
    while provider_id in existing_ids:
        ending = f"-{suffix}"
        provider_id = f"{base[: 80 - len(ending)]}{ending}"
        suffix += 1
    return provider_id


def _public_provider(provider_id: str) -> dict:
    public = _load_registry().public_config()
    for provider in public["providers"]:
        if provider["id"] == provider_id:
            return provider
    raise HTTPException(status_code=404, detail="Provider not found")


def _cache_provider_models(provider_id: str, models: list[str]) -> None:
    data = _load_local_providers()
    changed = False
    for provider in data["providers"]:
        if provider.get("id") == provider_id:
            provider["cachedModels"] = models
            changed = True
            break
    if changed:
        _save_local_providers(data)


def _pipeline_summary(registry_config: dict) -> dict:
    pipeline_id = registry_config["defaultPipelineId"]
    pipeline = next(
        item for item in registry_config["pipelines"] if item["id"] == pipeline_id
    )
    return {
        "image": {
            "provider": pipeline["image"]["providerId"],
            "model": pipeline["image"]["model"],
        },
        "review": {
            "provider": pipeline["review"]["providerId"],
            "model": pipeline["review"]["model"],
            "max_revisions": pipeline["maxRevisions"],
        },
        "questions": {
            "provider": pipeline["questions"]["providerId"],
            "model": pipeline["questions"]["model"],
        },
    }


def _normalize_gate_error_type(category: str) -> str:
    mapping = {
        "laterality": "LATERALITY",
        "text": "CHINESE_TEXT",
        "lesion_location": "LESION_LOCATION",
        "omission": "OMISSION",
        "anatomical_distortion": "ANATOMICAL_DISTORTION",
        "benign_malignant": "BENIGN_MALIGNANT",
        "suvmax": "SUVMAX",
        "style": "STYLE",
        "hallucination": "HALLUCINATION",
        "other": "OTHER",
    }
    return mapping.get(category, category.upper())


def _parse_reference_image_data_url(data_url: str | None) -> tuple[bytes | None, str]:
    if not data_url:
        return None, "image/png"
    match = REFERENCE_IMAGE_DATA_URL_PATTERN.fullmatch(data_url.strip())
    if not match:
        raise HTTPException(
            status_code=422,
            detail="Reference image must be a PNG, JPEG or WebP data URL.",
        )

    mime_type = match.group(1).lower()
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    try:
        image_bytes = base64.b64decode(match.group(2), validate=True)
    except binascii.Error as exc:
        raise HTTPException(
            status_code=422,
            detail="Reference image base64 data is invalid.",
        ) from exc
    if len(image_bytes) > REFERENCE_IMAGE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Reference image is larger than 15 MB.",
        )
    return image_bytes, mime_type


def _image_extension(mime_type: str) -> str:
    return {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
        mime_type,
        "png",
    )


def _gate_outcome(result) -> dict:
    attempt_count = len(result.attempts)
    revision_count = max(0, attempt_count - 1)
    if result.accepted is None:
        return {
            "status": "NOT_REVIEWED",
            "attemptCount": attempt_count,
            "revisionCount": revision_count,
            "returnedAttempt": attempt_count,
            "finalErrorTypes": [],
            "allErrorTypes": [],
        }
    all_error_types = []
    for attempt in result.attempts:
        if attempt.review:
            all_error_types.extend(
                _normalize_gate_error_type(issue.category)
                for issue in attempt.review.issues
            )
    final_review = result.attempts[-1].review
    final_error_types = (
        [_normalize_gate_error_type(issue.category) for issue in final_review.issues]
        if final_review and not final_review.passed
        else []
    )
    reason = final_review.summary if final_review else ""
    return {
        "status": "PASS" if result.accepted else "FAIL",
        "reason": reason,
        "attemptCount": attempt_count,
        "revisionCount": revision_count,
        "returnedAttempt": attempt_count,
        "finalErrorTypes": list(dict.fromkeys(final_error_types)),
        "allErrorTypes": list(dict.fromkeys(all_error_types)),
    }


async def _emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    status: str,
    message: str,
    **details: object,
) -> None:
    if callback is None:
        return
    await callback(
        {
            "type": "progress",
            "stage": stage,
            "status": status,
            "message": message,
            **details,
        }
    )


async def _build_generation_display_plan(
    *,
    payload: GeneratePayload,
    text_binding: ResolvedBinding,
    progress_callback: ProgressCallback | None,
) -> DisplayPlan | None:
    if payload.conclusion_text is None:
        return None

    finding_types, detail_fields = normalize_display_selection(
        payload.display_finding_types,
        payload.display_detail_fields,
    )
    await _emit_progress(
        progress_callback,
        stage="report_parsing",
        status="running",
        message="正在调用 AI 筛选需要展示的报告信息",
        providerLabel=text_binding.provider_label,
        model=text_binding.model,
    )
    planner = OpenAIDisplayPlanner(
        api_key=text_binding.api_key,
        model=text_binding.model,
        base_url=text_binding.base_url,
        timeout_seconds=text_binding.timeout_seconds,
        api_style=text_binding.api_style,
    )
    try:
        return await planner.plan(
            conclusion_text=payload.conclusion_text,
            findings_text=payload.findings_text,
            finding_types=list(finding_types),
            detail_fields=list(detail_fields),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=_upstream_error_detail("report_parsing", text_binding, exc),
        ) from exc


async def _build_generation_laterality_plan(
    *,
    report_text: str,
    mode: Literal["auto", "ai", "script"],
    laterality_binding: ResolvedBinding,
    progress_callback: ProgressCallback | None,
) -> LateralityPlan:
    await _emit_progress(
        progress_callback,
        stage="laterality",
        status="running",
        message="正在建立脚本左右清单",
    )
    script_plan = build_laterality_plan(report_text)
    if mode == "script":
        return script_plan

    if not laterality_binding.api_key:
        if mode == "ai":
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "PROVIDER_CONFIGURATION_ERROR",
                    "stage": "laterality",
                    "message": "AI 左右规划需要可用的文本模型 API Key。",
                    "providerId": laterality_binding.provider_id,
                    "suggestion": "请配置问题/文本供应商，或将 laterality_mode 改为 script。",
                },
            )
        return script_plan

    await _emit_progress(
        progress_callback,
        stage="laterality",
        status="running",
        message="正在调用 AI 转换结构化左右清单",
        providerLabel=laterality_binding.provider_label,
        model=laterality_binding.model,
    )
    planner = OpenAILateralityPlanner(
        api_key=laterality_binding.api_key,
        model=laterality_binding.model,
        base_url=laterality_binding.base_url,
        timeout_seconds=laterality_binding.timeout_seconds,
        api_style=laterality_binding.api_style,
    )
    try:
        ai_plan = await planner.plan(report_text)
    except Exception as exc:
        if mode == "ai":
            raise HTTPException(
                status_code=502,
                detail=_upstream_error_detail("laterality", laterality_binding, exc),
            ) from exc
        await _emit_progress(
            progress_callback,
            stage="laterality",
            status="running",
            message="AI 左右规划失败，已回退脚本左右清单",
        )
        return script_plan

    return merge_laterality_plans(ai_plan, script_plan)


def _upstream_error_detail(
    stage: str,
    binding: ResolvedBinding,
    exc: BaseException,
) -> dict:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        code = "UPSTREAM_AUTHENTICATION_FAILED"
        reason = "上游服务拒绝了 API 密钥。"
        suggestion = "请检查该阶段选择的供应商与 API Key 是否匹配。"
    elif status_code == 404:
        code = "UPSTREAM_NOT_FOUND"
        reason = "上游服务没有找到当前 API 路径。"
        suggestion = (
            "请检查供应商 API 模式和 Base URL。兼容网关通常应配置为 /v1 根地址，"
            "并使用其支持的接口类型。"
        )
    elif status_code == 429:
        code = "UPSTREAM_RATE_LIMITED"
        reason = "上游服务触发了额度或频率限制。"
        suggestion = "请检查账户额度，或稍后重试。"
    elif status_code and status_code >= 500:
        code = "UPSTREAM_SERVICE_ERROR"
        reason = "上游服务暂时不可用。"
        suggestion = "请稍后重试；若持续失败，请检查供应商状态。"
    else:
        code = "UPSTREAM_REQUEST_FAILED"
        reason = "上游请求未成功完成。"
        suggestion = "请检查该阶段的供应商、模型、Base URL 和网络连接。"
    return {
        "code": code,
        "stage": stage,
        "message": reason,
        "providerId": binding.provider_id,
        "providerLabel": binding.provider_label,
        "model": binding.model,
        "upstreamStatus": status_code,
        "suggestion": suggestion,
    }


def _configuration_error_detail(exc: ValueError) -> dict:
    raw_message = str(exc)
    provider_match = re.search(r"provider '([^']+)'", raw_message)
    provider_id = provider_match.group(1) if provider_match else None
    if "requires a Base URL" in raw_message:
        message = f"供应商“{provider_id or '当前供应商'}”缺少 Base URL。"
    elif "requires environment variable" in raw_message:
        message = f"供应商“{provider_id or '当前供应商'}”缺少可用的 API Key。"
    else:
        message = "当前供应商或模型配置无效。"
    return {
        "code": "PROVIDER_CONFIGURATION_ERROR",
        "stage": "configuration",
        "message": message,
        "providerId": provider_id,
        "suggestion": "请在 API 设置中补全对应供应商的 API Key、Base URL 和模型。",
    }


def _safe_case_folder_name(case_id: str, run_id: str) -> str:
    folder_name = INVALID_CASE_PATH_CHARS.sub("_", case_id.strip()).rstrip(" .")
    if not folder_name:
        folder_name = f"case_{run_id[:8]}"
    if folder_name.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        folder_name = f"_{folder_name}"
    return folder_name


def _build_case_run_directory(
    case_id: str,
    run_id: str,
    storage_time: datetime | None = None,
) -> Path:
    current_time = storage_time or datetime.now().astimezone()
    date_folder = current_time.date().isoformat()
    case_folder = _safe_case_folder_name(case_id, run_id)
    return RUNTIME_DIR / date_folder / case_folder / run_id


def _case_directory(run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    legacy_case_dir = RUNTIME_DIR / run_id
    if legacy_case_dir.is_dir():
        return legacy_case_dir
    for case_dir in RUNTIME_DIR.glob(f"*/*/{run_id}"):
        if case_dir.is_dir():
            return case_dir
    raise HTTPException(status_code=404, detail="Run not found")


def _load_run_manifest(run_id: str) -> tuple[Path, dict]:
    case_dir = _case_directory(run_id)
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Run manifest is invalid") from exc
    return case_dir, manifest


def _final_attempt_image_name(manifest: dict) -> str | None:
    attempts = manifest.get("attempts") or []
    for attempt in reversed(attempts):
        if isinstance(attempt, dict) and attempt.get("image"):
            return str(attempt["image"])
    return None


def _image_mime_type(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def _run_file_path(case_dir: Path, filename: str) -> Path:
    if not RUN_FILE_PATTERN.fullmatch(filename) or filename == "manifest.json":
        raise HTTPException(status_code=404, detail="File not found")
    root = case_dir.resolve()
    file_path = (case_dir / filename).resolve()
    try:
        file_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return file_path


def _run_image_url(run_id: str, manifest: dict) -> str | None:
    image_name = _final_attempt_image_name(manifest)
    if not image_name:
        return None
    return f"/api/runs/{run_id}/files/{image_name}"


def _public_run_manifest(
    case_dir: Path,
    manifest: dict,
    *,
    include_image_data_url: bool,
) -> dict:
    public_manifest = dict(manifest)
    run_id = str(public_manifest.get("run_id") or case_dir.name)
    image_name = _final_attempt_image_name(public_manifest)
    public_manifest["image_url"] = _run_image_url(run_id, public_manifest)
    if include_image_data_url and image_name:
        image_path = _run_file_path(case_dir, image_name)
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        public_manifest["image_data_url"] = (
            f"data:{_image_mime_type(image_name)};base64,{image_data}"
        )
    return public_manifest


def _history_item(case_dir: Path, manifest: dict) -> dict:
    run_id = str(manifest.get("run_id") or case_dir.name)
    attempts = manifest.get("attempts") or []
    pipeline = manifest.get("pipeline") or {}
    gate_outcome = manifest.get("gate_outcome") or {}
    return {
        "run_id": run_id,
        "case_id": manifest.get("case_id") or "",
        "generated_at": manifest.get("generated_at"),
        "duration_seconds": manifest.get("duration_seconds"),
        "strategy": manifest.get("strategy"),
        "gate_enabled": manifest.get("gate_enabled"),
        "accepted": manifest.get("accepted"),
        "gate_outcome": gate_outcome,
        "attempt_count": len(attempts),
        "pipeline_label": pipeline.get("label"),
        "image_url": _run_image_url(run_id, manifest),
        "human_evaluation": manifest.get("human_evaluation"),
    }


app = FastAPI(title="PET-CT Patient Visualization", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def public_config():
    registry_config = _load_registry().public_config()
    return {
        **_pipeline_summary(registry_config),
        "providerRegistry": registry_config,
    }


@app.get("/api/providers")
async def list_providers():
    return _load_registry().public_config()


@app.post("/api/providers")
async def upsert_provider(payload: ProviderPayload):
    data = _load_local_providers()
    registry = _load_registry()
    base_provider = registry.get_provider(payload.id) if payload.id else None
    existing = {
        provider.get("id"): provider
        for provider in data["providers"]
        if provider.get("id")
    }
    provider_id = payload.id or _provider_id_from_label(
        payload.label,
        set(existing.keys()),
    )
    if not PROVIDER_ID_PATTERN.fullmatch(provider_id):
        raise HTTPException(
            status_code=422,
            detail="Provider ID can only contain letters, numbers, dots, dashes and underscores.",
        )

    previous = existing.get(provider_id, {})
    entry = {
        "id": provider_id,
        "label": payload.label.strip(),
        "apiStyle": payload.api_style,
        "timeoutSeconds": payload.timeout_seconds,
        "cachedModels": payload.cached_models or previous.get("cachedModels", []),
    }
    task_models = {
        key: value.strip()
        for key, value in payload.task_models.items()
        if key in {"image", "review", "questions"} and value.strip()
    }
    if task_models:
        entry["taskModels"] = task_models
    elif previous.get("taskModels"):
        entry["taskModels"] = previous["taskModels"]
    if base_provider and base_provider.api_key_env:
        entry["apiKeyEnv"] = base_provider.api_key_env
    if base_provider and base_provider.base_url_env:
        entry["baseUrlEnv"] = base_provider.base_url_env
    if payload.base_url and payload.base_url.strip():
        entry["baseUrl"] = payload.base_url.strip().rstrip("/")
    elif previous.get("baseUrl"):
        entry["baseUrl"] = previous["baseUrl"]
    elif base_provider and base_provider.base_url:
        entry["baseUrl"] = base_provider.base_url
    if payload.homepage and payload.homepage.strip():
        entry["homepage"] = payload.homepage.strip()
    elif previous.get("homepage"):
        entry["homepage"] = previous["homepage"]
    elif base_provider and base_provider.homepage:
        entry["homepage"] = base_provider.homepage
    if payload.notes and payload.notes.strip():
        entry["notes"] = payload.notes.strip()
    elif previous.get("notes"):
        entry["notes"] = previous["notes"]
    if payload.api_key and payload.api_key.strip():
        entry["apiKey"] = payload.api_key.strip()
    elif previous.get("apiKey"):
        entry["apiKey"] = previous["apiKey"]

    updated = False
    for index, provider in enumerate(data["providers"]):
        if provider.get("id") == provider_id:
            data["providers"][index] = entry
            updated = True
            break
    if not updated:
        data["providers"].append(entry)
    _save_local_providers(data)
    return {"provider": _public_provider(provider_id)}


@app.delete("/api/providers/{provider_id}")
async def delete_provider(provider_id: str):
    if not PROVIDER_ID_PATTERN.fullmatch(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found")
    data = _load_local_providers()
    before = len(data["providers"])
    data["providers"] = [
        provider
        for provider in data["providers"]
        if provider.get("id") != provider_id
    ]
    if len(data["providers"]) == before:
        raise HTTPException(status_code=404, detail="Local provider not found")
    _save_local_providers(data)
    return {"deleted": provider_id}


@app.post("/api/providers/fetch-models")
async def fetch_provider_models(payload: FetchModelsPayload):
    registry = _load_registry()
    provider = (
        registry.get_provider(payload.provider_id)
        if payload.provider_id
        else None
    )
    api_key = (payload.api_key or "").strip()
    base_url = (payload.base_url or "").strip()
    timeout_seconds = payload.timeout_seconds

    if provider:
        api_key = api_key or provider.resolved_api_key()
        base_url = base_url or (provider.resolved_base_url() or "")
        timeout_seconds = timeout_seconds or provider.timeout_seconds
    if not api_key:
        raise HTTPException(status_code=422, detail="API key is required.")

    try:
        models, source_url = await fetch_openai_compatible_models(
            api_key=api_key,
            base_url=base_url or None,
            timeout_seconds=timeout_seconds or 60,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    cache_provider_id = payload.cache_provider_id or (
        payload.provider_id if payload.provider_id in registry.local_provider_ids else None
    )
    if cache_provider_id:
        _cache_provider_models(cache_provider_id, models)
    return {
        "models": models,
        "count": len(models),
        "sourceUrl": source_url,
    }


@app.post("/api/generate")
async def generate(payload: GeneratePayload):
    return await _execute_generation(payload)


async def _execute_generation(
    payload: GeneratePayload,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    started_at = time.perf_counter()
    await _emit_progress(
        progress_callback,
        stage="configuration",
        status="running",
        message="正在检查供应商与模型配置",
    )
    registry = _load_registry()
    reference_image, reference_mime_type = _parse_reference_image_data_url(
        payload.reference_image_data_url
    )
    review_strength = _effective_review_strength(payload)
    review_enabled = review_strength != ReviewStrength.OFF
    uses_structured_input = payload.conclusion_text is not None
    try:
        pipeline_config = registry.resolve_pipeline(
            payload.pipeline_id,
            image_override=payload.image_override,
            review_override=payload.review_override,
            question_override=payload.question_override,
            max_revisions=payload.max_revisions,
            require_review=review_enabled,
            require_questions=(
                payload.generate_questions
                or payload.laterality_mode == "ai"
                or uses_structured_input
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=_configuration_error_detail(exc),
        ) from exc
    await _emit_progress(
        progress_callback,
        stage="configuration",
        status="completed",
        message="供应商与模型配置检查完成",
        pipelineId=pipeline_config.id,
    )
    await _emit_progress(
        progress_callback,
        stage="report_parsing",
        status="running",
        message="正在解析报告关键发现",
    )
    display_plan = await _build_generation_display_plan(
        payload=payload,
        text_binding=pipeline_config.questions,
        progress_callback=progress_callback,
    )
    await _emit_progress(
        progress_callback,
        stage="report_parsing",
        status="completed",
        message="报告解析完成" if display_plan is None else "展示信息筛选完成",
        reportLength=len(payload.report_text),
        displayItemCount=len(display_plan.drawable_items()) if display_plan else 0,
    )
    if display_plan:
        await _emit_progress(
            progress_callback,
            stage="laterality",
            status="running",
            message="正在根据展示计划固定左右清单",
        )
        laterality_plan = build_laterality_plan_from_display_plan(display_plan)
    else:
        laterality_plan = await _build_generation_laterality_plan(
            report_text=payload.report_text,
            mode=payload.laterality_mode,
            laterality_binding=pipeline_config.questions,
            progress_callback=progress_callback,
        )
    await _emit_progress(
        progress_callback,
        stage="laterality",
        status="completed",
        message=(
            f"左右规划完成：{len(laterality_plan.findings)} 项"
            f"（{laterality_plan.source}）"
        ),
        findingCount=len(laterality_plan.findings),
        source=laterality_plan.source,
    )

    image_options = pipeline_config.image.options
    image_generator = OpenAIImageGenerator(
        api_key=pipeline_config.image.api_key,
        model=pipeline_config.image.model,
        base_url=pipeline_config.image.base_url,
        size=image_options.get("size", "1536x1024"),
        quality=image_options.get("quality", "medium"),
        provider_id=pipeline_config.image.provider_id,
        timeout_seconds=pipeline_config.image.timeout_seconds,
        options=image_options,
    )
    reviewer = None
    if review_enabled:
        reviewer = OpenAIVisualReviewer(
            api_key=pipeline_config.review.api_key,
            model=pipeline_config.review.model,
            base_url=pipeline_config.review.base_url,
            timeout_seconds=pipeline_config.review.timeout_seconds,
            api_style=pipeline_config.review.api_style,
            review_strength=review_strength,
        )

    prompt_template = PET_CT_IMG2IMG_PROMPT if reference_image else PET_CT_VISUALIZATION_PROMPT
    display_prompt_block = (
        f"{display_plan.to_prompt_block()}\n\n"
        if display_plan
        else ""
    )
    prompt = (
        f"{display_prompt_block}"
        f"{laterality_plan.to_prompt_block()}\n\n"
        f"{prompt_template.format(report_content=payload.report_text)}"
    )

    request = GenerationRequest(
        case_id=payload.case_id,
        report_text=payload.report_text,
        prompt=prompt,
        reference_image=reference_image,
        reference_mime_type=reference_mime_type,
        laterality_plan=laterality_plan,
        display_plan=display_plan,
    )
    generation_pipeline = GenerationPipeline(
        image_generator,
        reviewer,
        max_revisions=pipeline_config.max_revisions,
    )

    async def run_questions() -> list[dict]:
        await _emit_progress(
            progress_callback,
            stage="questions",
            status="running",
            message="正在生成患者理解度测试题",
            providerLabel=pipeline_config.questions.provider_label,
            model=pipeline_config.questions.model,
        )
        question_service = OpenAIQuestionService(
            api_key=pipeline_config.questions.api_key,
            model=pipeline_config.questions.model,
            base_url=pipeline_config.questions.base_url,
            timeout_seconds=pipeline_config.questions.timeout_seconds,
            api_style=pipeline_config.questions.api_style,
        )
        try:
            generated_questions = await question_service.generate(payload.report_text)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=_upstream_error_detail(
                    "questions",
                    pipeline_config.questions,
                    exc,
                ),
            ) from exc
        await _emit_progress(
            progress_callback,
            stage="questions",
            status="completed",
            message=f"已生成 {len(generated_questions)} 道测试题",
            questionCount=len(generated_questions),
        )
        return generated_questions

    questions_task: asyncio.Task[list[dict]] | None = None
    if payload.generate_questions:
        questions_task = asyncio.create_task(run_questions())
        await asyncio.sleep(0)
    else:
        await _emit_progress(
            progress_callback,
            stage="questions",
            status="skipped",
            message="未启用患者理解度测试题",
        )

    async def pipeline_progress(event: dict[str, object]) -> None:
        if progress_callback is None:
            return
        binding = (
            pipeline_config.review
            if event.get("stage") == "review"
            else pipeline_config.image
        )
        await progress_callback(
            {
                **event,
                "providerLabel": binding.provider_label,
                "model": binding.model,
            }
        )

    try:
        result = await generation_pipeline.run(
            request,
            gate_enabled=review_enabled,
            progress_callback=pipeline_progress if progress_callback else None,
        )
    except PipelineStageError as exc:
        if questions_task:
            if not questions_task.done():
                questions_task.cancel()
            try:
                await questions_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        binding = (
            pipeline_config.review
            if exc.stage == "review"
            else pipeline_config.image
        )
        cause = exc.__cause__ or exc
        raise HTTPException(
            status_code=502,
            detail=_upstream_error_detail(exc.stage, binding, cause),
        ) from exc

    if not review_enabled:
        await _emit_progress(
            progress_callback,
            stage="review",
            status="skipped",
            message="未启用 AI 图片审查",
        )

    questions = await questions_task if questions_task else []

    await _emit_progress(
        progress_callback,
        stage="saving",
        status="running",
        message="正在保存图片与运行记录",
    )
    try:
        run_id = uuid.uuid4().hex
        case_dir = _build_case_run_directory(payload.case_id, run_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        reference_image_name = None
        if reference_image:
            reference_image_name = f"reference.{_image_extension(reference_mime_type)}"
            (case_dir / reference_image_name).write_bytes(reference_image)
        attempts_manifest = []
        for index, attempt in enumerate(result.attempts, start=1):
            extension = (
                "jpg"
                if attempt.image.mime_type == "image/jpeg"
                else attempt.image.mime_type.split("/")[-1]
            )
            image_name = f"attempt_{index}.{extension}"
            (case_dir / image_name).write_bytes(attempt.image.content)
            attempts_manifest.append(
                {
                    "image": image_name,
                    "provider": attempt.image.provider,
                    "model": attempt.image.model,
                    "review": asdict(attempt.review) if attempt.review else None,
                }
            )

        gate_outcome = _gate_outcome(result)
        generated_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = round(time.perf_counter() - started_at, 2)
        reproducibility = build_reproducibility_metadata(
            project_root=BASE_DIR,
            image_prompt_template=prompt_template,
            rendered_image_prompt=prompt,
            random_seed=payload.random_seed,
            reference_image=reference_image,
        )
        selected_finding_types, selected_detail_fields = normalize_display_selection(
            payload.display_finding_types,
            payload.display_detail_fields,
        )
        manifest = {
            "run_id": run_id,
            "generated_at": generated_at,
            "duration_seconds": duration_seconds,
            "pairing_key": build_record_key(payload.case_id, payload.report_text),
            "case_id": payload.case_id,
            "report_text": payload.report_text,
            "input_text": {
                "conclusion": payload.conclusion_text,
                "findings": payload.findings_text,
                "derived_report_text": payload.report_text,
            },
            "display_selection": {
                "finding_types": [item.value for item in selected_finding_types],
                "detail_fields": [item.value for item in selected_detail_fields],
            },
            "display_plan": display_plan.to_manifest() if display_plan else None,
            "experiment": {
                "id": payload.experiment_id,
                "item_id": payload.experiment_item_id,
                "cancer_type": payload.cancer_type,
            },
            "reproducibility": reproducibility,
            "strategy": "GATED" if review_enabled else "UNGATED",
            "gate_enabled": review_enabled,
            "requested_gate_enabled": payload.gate_enabled,
            "review_strength": review_strength.value,
            "reference_image": reference_image_name,
            "accepted": result.accepted,
            "gate_outcome": gate_outcome,
            "laterality_plan": laterality_plan.to_manifest(),
            "pipeline": {
                "id": pipeline_config.id,
                "label": pipeline_config.label,
                "image": {
                    "provider": pipeline_config.image.provider_id,
                    "provider_label": pipeline_config.image.provider_label,
                    "api_style": pipeline_config.image.api_style,
                    "base_url": pipeline_config.image.base_url,
                    "timeout_seconds": pipeline_config.image.timeout_seconds,
                    "model": pipeline_config.image.model,
                    "options": pipeline_config.image.options,
                },
                "review": {
                    "provider": pipeline_config.review.provider_id,
                    "provider_label": pipeline_config.review.provider_label,
                    "api_style": pipeline_config.review.api_style,
                    "base_url": pipeline_config.review.base_url,
                    "timeout_seconds": pipeline_config.review.timeout_seconds,
                    "model": pipeline_config.review.model,
                },
                "questions": {
                    "provider": pipeline_config.questions.provider_id,
                    "provider_label": pipeline_config.questions.provider_label,
                    "api_style": pipeline_config.questions.api_style,
                    "base_url": pipeline_config.questions.base_url,
                    "timeout_seconds": pipeline_config.questions.timeout_seconds,
                    "model": pipeline_config.questions.model,
                },
                "max_revisions": pipeline_config.max_revisions,
            },
            "attempts": attempts_manifest,
            "questions": questions,
            "human_evaluation": None,
        }
        (case_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RESULT_SAVE_FAILED",
                "stage": "saving",
                "message": "图片已经生成，但保存运行记录失败。",
                "suggestion": "请检查 runtime/cases 目录的写入权限和可用空间。",
            },
        ) from exc

    await _emit_progress(
        progress_callback,
        stage="saving",
        status="completed",
        message="图片与运行记录已保存",
        runId=run_id,
    )
    image_data = base64.b64encode(result.final_image.content).decode("ascii")
    return {
        **manifest,
        "image_data_url": f"data:{result.final_image.mime_type};base64,{image_data}",
    }


@app.post("/api/generate/stream")
async def generate_stream(payload: GeneratePayload):
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def progress_callback(event: dict[str, object]) -> None:
        await queue.put(event)

    async def worker() -> None:
        try:
            data = await _execute_generation(payload, progress_callback)
        except HTTPException as exc:
            detail = (
                exc.detail
                if isinstance(exc.detail, dict)
                else {
                    "code": "REQUEST_FAILED",
                    "stage": "unknown",
                    "message": str(exc.detail),
                }
            )
            await queue.put(
                {
                    "type": "error",
                    "httpStatus": exc.status_code,
                    "error": detail,
                }
            )
        except Exception:
            await queue.put(
                {
                    "type": "error",
                    "httpStatus": 500,
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "stage": "unknown",
                        "message": "服务内部发生未预期错误。",
                        "suggestion": "请查看服务日志，或重启服务后重试。",
                    },
                }
            )
        else:
            await queue.put({"type": "result", "data": data})

    task = asyncio.create_task(worker())

    async def stream_events():
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in {"result", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history")
async def list_history(limit: int = 50, case_id: str | None = None):
    bounded_limit = min(max(limit, 1), 200)
    case_filter = case_id.strip() if case_id else ""
    items: list[dict] = []
    if RUNTIME_DIR.exists():
        for manifest_path in RUNTIME_DIR.glob("**/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if case_filter and manifest.get("case_id") != case_filter:
                continue
            items.append(_history_item(manifest_path.parent, manifest))
    items.sort(key=lambda item: item.get("generated_at") or "", reverse=True)
    return {"items": items[:bounded_limit], "count": len(items[:bounded_limit])}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    case_dir, manifest = _load_run_manifest(run_id)
    return _public_run_manifest(case_dir, manifest, include_image_data_url=True)


@app.get("/api/runs/{run_id}/files/{filename}")
async def get_run_file(run_id: str, filename: str):
    case_dir = _case_directory(run_id)
    file_path = _run_file_path(case_dir, filename)
    return FileResponse(file_path, media_type=_image_mime_type(filename))


@app.put("/api/runs/{run_id}/evaluation")
async def save_evaluation(run_id: str, evaluation: HumanEvaluation):
    case_dir = _case_directory(run_id)
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["human_evaluation"] = evaluation.model_dump(mode="json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "human_evaluation": manifest["human_evaluation"],
    }


def main():
    import uvicorn

    host = os.getenv("PETCT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("PETCT_WEB_PORT", "8000"))
    uvicorn.run("webapp.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
