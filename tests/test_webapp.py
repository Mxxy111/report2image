import asyncio
import base64
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import webapp.main as webapp_main
from petct.evaluation import ErrorType, HumanEvaluation, OverallDecision
from petct.display_plan import (
    DisplayDetailField,
    DisplayFindingType,
    DisplayPlan,
    DisplayPlanItem,
    ReviewStrength,
)
from petct.generation import (
    GeneratedImage,
    GenerationAttempt,
    PipelineResult,
    ReviewDecision,
    ReviewIssue,
)
from petct.laterality import (
    CanvasSide,
    LateralityFinding,
    LateralityPlan,
    PatientSide,
)
from petct.provider_config import ResolvedBinding
from webapp.main import (
    GeneratePayload,
    _execute_generation,
    _gate_outcome,
    _parse_reference_image_data_url,
    _upstream_error_detail,
    generate_stream,
    get_run,
    get_run_file,
    health,
    list_history,
    public_config,
    save_evaluation,
)
from webapp.main import (
    FetchModelsPayload,
    ProviderPayload,
    delete_provider,
    fetch_provider_models,
    upsert_provider,
)


def test_health_endpoint_payload():
    assert asyncio.run(health()) == {"status": "ok"}


def test_index_loads_the_application_script():
    html = (webapp_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert '<script src="/static/app.js"></script>' in html
    assert "scrilt" not in html
    assert "/static/all.js" not in html


def test_provider_api_style_select_uses_valid_values():
    html = (webapp_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'value="openai_compatible"' in html
    assert 'value="openai"' in html
    assert "olenai" not in html
    assert "comlatible" not in html


def test_static_css_preserves_hidden_attribute_for_preview_swap():
    css = (webapp_main.STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "[hidden] { display: none !important; }" in css


def test_print_report_hides_internal_quality_sections_and_shows_generation_timing():
    html = (webapp_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (webapp_main.STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    app_js = (webapp_main.STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="printGeneratedAt"' in html
    assert 'id="printDuration"' in html
    assert "#gateSection" in css
    assert "#humanEvaluationSection" in css
    assert "#gateBadge" in css
    assert "formatDuration" in app_js
    assert "duration_seconds" in app_js


def test_progress_ui_includes_report_parsing_and_laterality_steps():
    html = (webapp_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (webapp_main.STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'data-stage="report_parsing"' in html
    assert 'data-stage="laterality"' in html
    assert 'report_parsing: "信息筛选"' in app_js
    assert 'laterality: "左右规划"' in app_js


def test_history_ui_has_dialog_and_uses_history_api():
    html = (webapp_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (webapp_main.STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    app_js = (webapp_main.STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="historyButton"' in html
    assert 'id="historyDialog"' in html
    assert "history-list" in css
    assert 'fetch(`/api/history?' in app_js
    assert 'fetch(`/api/runs/${runId}`)' in app_js
    assert "renderRunResult(data)" in app_js


def test_laterality_planner_is_shown_as_sharing_question_model_config():
    html = (webapp_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (webapp_main.STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "出题/左右规划供应商" in html
    assert "出题/左右规划模型" in html
    assert '["左右规划", effective.questions]' in app_js


def test_frontend_exposes_display_plan_inputs_and_review_strength():
    html = (webapp_main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (webapp_main.STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="conclusionText"' in html
    assert 'id="findingsText"' in html
    assert 'id="reviewStrengthSelect"' in html
    assert 'value="MALIGNANT"' in html
    assert 'value="SUVMAX"' in html
    assert "conclusion_text: conclusionText" in app_js
    assert "findings_text: findingsText || null" in app_js
    assert "display_finding_types: selectedCheckboxValues" in app_js
    assert "display_detail_fields: selectedCheckboxValues" in app_js
    assert "review_strength: reviewStrength" in app_js


def test_config_endpoint_does_not_expose_api_keys():
    payload = asyncio.run(public_config())
    assert payload["providerRegistry"]["defaultPipelineId"] == "split_default"
    assert len(payload["providerRegistry"]["pipelines"]) >= 3
    assert all(
        "apiKey" not in provider
        for provider in payload["providerRegistry"]["providers"]
    )


def test_upsert_provider_persists_local_provider_without_exposing_key(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp_main, "LOCAL_PROVIDER_CONFIG", tmp_path / "local_providers.json")
    payload = ProviderPayload(
        label="SiliconFlow",
        apiKey="secret-key",
        baseUrl="https://api.siliconflow.cn/v1",
        taskModels={
            "image": "gpt-image-2",
            "review": "moonshotai/Kimi-K2",
            "questions": "deepseek-ai/DeepSeek-V3.2",
        },
    )

    response = asyncio.run(upsert_provider(payload))

    saved = json.loads((tmp_path / "local_providers.json").read_text(encoding="utf-8"))
    assert response["provider"]["label"] == "SiliconFlow"
    assert response["provider"]["isConfigured"] is True
    assert saved["providers"][0]["apiKey"] == "secret-key"
    assert saved["providers"][0]["taskModels"]["image"] == "gpt-image-2"
    assert "secret-key" not in json.dumps(response)


def test_upsert_builtin_provider_preserves_environment_key_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp_main, "LOCAL_PROVIDER_CONFIG", tmp_path / "local_providers.json")
    monkeypatch.setenv("PETCT_QUESTION_API_KEY", "env-secret")
    payload = ProviderPayload(
        id="question_api",
        label="Question API",
        apiKey=None,
        baseUrl="https://api.example.com/v1",
    )

    response = asyncio.run(upsert_provider(payload))

    saved = json.loads((tmp_path / "local_providers.json").read_text(encoding="utf-8"))
    assert saved["providers"][0]["apiKeyEnv"] == "PETCT_QUESTION_API_KEY"
    assert response["provider"]["isConfigured"] is True
    assert "env-secret" not in json.dumps(response)


def test_delete_provider_only_removes_local_profiles(tmp_path, monkeypatch):
    local_path = tmp_path / "local_providers.json"
    local_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": [
                    {
                        "id": "local-one",
                        "label": "Local One",
                        "apiStyle": "openai_compatible",
                        "apiKey": "secret",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_main, "LOCAL_PROVIDER_CONFIG", local_path)

    response = asyncio.run(delete_provider("local-one"))

    saved = json.loads(local_path.read_text(encoding="utf-8"))
    assert response["deleted"] == "local-one"
    assert saved["providers"] == []


def test_fetch_models_uses_saved_provider_and_caches_result(tmp_path, monkeypatch):
    local_path = tmp_path / "local_providers.json"
    local_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": [
                    {
                        "id": "local-one",
                        "label": "Local One",
                        "apiStyle": "openai_compatible",
                        "apiKey": "secret",
                        "baseUrl": "https://provider.example/v1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_main, "LOCAL_PROVIDER_CONFIG", local_path)

    async def fake_fetch_models(**kwargs):
        assert kwargs["api_key"] == "secret"
        assert kwargs["base_url"] == "https://provider.example/v1"
        return ["model-a", "model-b"], "https://provider.example/v1/models"

    monkeypatch.setattr(webapp_main, "fetch_openai_compatible_models", fake_fetch_models)

    response = asyncio.run(
        fetch_provider_models(
            FetchModelsPayload(providerId="local-one", cacheProviderId="local-one")
        )
    )

    saved = json.loads(local_path.read_text(encoding="utf-8"))
    assert response["models"] == ["model-a", "model-b"]
    assert saved["providers"][0]["cachedModels"] == ["model-a", "model-b"]


def test_generate_payload_rejects_short_report():
    with pytest.raises(ValidationError):
        GeneratePayload(
            case_id="P001",
            report_text="太短",
            gate_enabled=False,
            generate_questions=False,
        )


def test_generate_payload_accepts_structured_report_inputs_and_derives_report_text():
    payload = GeneratePayload(
        case_id="P001",
        conclusion_text="右肾占位，考虑恶性。",
        findings_text="右肾见软组织肿块，约3.2 cm，SUVmax=12.9。",
        gate_enabled=False,
        generate_questions=False,
    )

    assert payload.report_text == (
        "检查结论：\n右肾占位，考虑恶性。\n\n"
        "检查所见：\n右肾见软组织肿块，约3.2 cm，SUVmax=12.9。"
    )


def test_parse_reference_image_data_url_accepts_supported_image():
    raw = b"fake-png"
    encoded = base64.b64encode(raw).decode("ascii")

    image_bytes, mime_type = _parse_reference_image_data_url(
        f"data:image/png;base64,{encoded}"
    )

    assert image_bytes == raw
    assert mime_type == "image/png"


def test_parse_reference_image_data_url_rejects_non_image_data_url():
    with pytest.raises(HTTPException) as exc_info:
        _parse_reference_image_data_url("data:text/plain;base64,SGVsbG8=")

    assert exc_info.value.status_code == 422


def test_build_case_run_directory_groups_runs_by_date_and_case_id(tmp_path, monkeypatch):
    run_id = "a" * 32
    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)

    case_dir = webapp_main._build_case_run_directory(
        "P/001",
        run_id,
        datetime(2026, 6, 21, 9, 30, tzinfo=timezone.utc),
    )

    assert case_dir == tmp_path / "2026-06-21" / "P_001" / run_id


def test_execute_generation_records_duration_metadata(tmp_path, monkeypatch):
    binding = ResolvedBinding(
        provider_id="fake_provider",
        provider_label="Fake Provider",
        api_style="openai_compatible",
        api_key="secret",
        base_url="https://provider.example/v1",
        timeout_seconds=300,
        model="fake-model",
        options={},
    )

    class FakeRegistry:
        def resolve_pipeline(self, *args, **kwargs):
            return SimpleNamespace(
                id="fake_pipeline",
                label="Fake Pipeline",
                image=binding,
                review=binding,
                questions=binding,
                max_revisions=0,
            )

    class FakeGenerationPipeline:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, request, *, gate_enabled, progress_callback=None):
            assert request.laterality_plan is not None
            assert "结构化左右清单" in request.prompt
            assert "右肾" in request.prompt
            image = GeneratedImage(b"fake-image", "image/png", "fake_provider", "fake-model")
            return PipelineResult(
                final_image=image,
                accepted=None,
                gate_enabled=False,
                attempts=(GenerationAttempt(image=image, review=None),),
            )

    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(webapp_main, "_load_registry", lambda: FakeRegistry())
    monkeypatch.setattr(webapp_main, "GenerationPipeline", FakeGenerationPipeline)
    progress_events = []

    async def capture_progress(event):
        progress_events.append(event)

    response = asyncio.run(
        _execute_generation(
            GeneratePayload(
                case_id="P001",
                report_text="右肾见占位性病变，考虑恶性。",
                gate_enabled=False,
                generate_questions=False,
            ),
            capture_progress,
        )
    )

    manifest_paths = list(
        tmp_path.glob(f"????-??-??/P001/{response['run_id']}/manifest.json")
    )
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert response["generated_at"] == manifest["generated_at"]
    assert response["duration_seconds"] == manifest["duration_seconds"]
    assert response["duration_seconds"] >= 0
    assert manifest["reproducibility"]["prompts"]["image"]["version"]
    assert len(manifest["reproducibility"]["prompts"]["image"]["sha256"]) == 64
    assert len(manifest["reproducibility"]["software"]["sourceTreeSha256"]) == 64
    assert manifest["reproducibility"]["randomSeed"] is None
    assert manifest["reproducibility"]["providerSeedApplied"] is False
    assert manifest["laterality_plan"]["findings"][0]["patient_side"] == "RIGHT"
    assert manifest["laterality_plan"]["findings"][0]["canvas_side"] == "LEFT"
    assert "report_parsing" in [event["stage"] for event in progress_events]
    assert "laterality" in [event["stage"] for event in progress_events]


def test_execute_generation_uses_display_plan_for_structured_inputs(
    tmp_path,
    monkeypatch,
):
    binding = ResolvedBinding(
        provider_id="fake_provider",
        provider_label="Fake Provider",
        api_style="openai_compatible",
        api_key="secret",
        base_url="https://provider.example/v1",
        timeout_seconds=300,
        model="fake-model",
        options={},
    )
    captured = {}

    class FakeRegistry:
        def resolve_pipeline(self, *args, **kwargs):
            captured["require_questions"] = kwargs["require_questions"]
            return SimpleNamespace(
                id="fake_pipeline",
                label="Fake Pipeline",
                image=binding,
                review=binding,
                questions=binding,
                max_revisions=0,
            )

    class FakeDisplayPlanner:
        def __init__(self, **kwargs):
            captured["display_model"] = kwargs["model"]

        async def plan(
            self,
            *,
            conclusion_text,
            findings_text=None,
            finding_types=None,
            detail_fields=None,
        ):
            captured["display_inputs"] = {
                "conclusion": conclusion_text,
                "findings": findings_text,
                "finding_types": tuple(finding_types),
                "detail_fields": tuple(detail_fields),
            }
            return DisplayPlan(
                selectedFindingTypes=[DisplayFindingType.MALIGNANT],
                selectedDetailFields=[
                    DisplayDetailField.SUVMAX,
                    DisplayDetailField.LESION_SIZE,
                ],
                items=[
                    DisplayPlanItem(
                        id="item-1",
                        contentTypes=[DisplayFindingType.MALIGNANT],
                        priority="primary",
                        anatomy="右肾",
                        patientSide=PatientSide.RIGHT,
                        nature="考虑恶性",
                        colorClass="malignant_suspected",
                        labelText="右肾占位，SUVmax 12.9，约3.2 cm",
                        suvmax="12.9",
                        size="3.2 cm",
                        conclusionEvidence="右肾占位，考虑恶性",
                        findingsEvidence="右肾见软组织肿块，约3.2 cm，SUVmax=12.9",
                    )
                ],
            )

    class FakeGenerationPipeline:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, request, *, gate_enabled, progress_callback=None):
            captured["prompt"] = request.prompt
            captured["display_plan"] = request.display_plan
            captured["laterality_plan"] = request.laterality_plan
            image = GeneratedImage(b"fake-image", "image/png", "fake_provider", "fake-model")
            return PipelineResult(
                final_image=image,
                accepted=None,
                gate_enabled=False,
                attempts=(GenerationAttempt(image=image, review=None),),
            )

    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(webapp_main, "_load_registry", lambda: FakeRegistry())
    monkeypatch.setattr(webapp_main, "OpenAIDisplayPlanner", FakeDisplayPlanner)
    monkeypatch.setattr(webapp_main, "GenerationPipeline", FakeGenerationPipeline)

    response = asyncio.run(
        _execute_generation(
            GeneratePayload(
                case_id="P001",
                conclusion_text="右肾占位，考虑恶性。",
                findings_text="右肾见软组织肿块，约3.2 cm，SUVmax=12.9。",
                display_finding_types=[DisplayFindingType.MALIGNANT],
                display_detail_fields=[
                    DisplayDetailField.SUVMAX,
                    DisplayDetailField.LESION_SIZE,
                ],
                gate_enabled=False,
                generate_questions=False,
            )
        )
    )

    assert captured["require_questions"] is True
    assert captured["display_model"] == "fake-model"
    assert captured["display_inputs"]["finding_types"] == (DisplayFindingType.MALIGNANT,)
    assert "展示计划" in captured["prompt"]
    assert "右肾占位，SUVmax 12.9" in captured["prompt"]
    assert captured["display_plan"].items[0].suvmax == "12.9"
    assert captured["laterality_plan"].source == "display_plan"
    assert response["display_plan"]["items"][0]["suvmax"] == "12.9"
    assert response["input_text"]["conclusion"] == "右肾占位，考虑恶性。"


def test_execute_generation_uses_ai_laterality_planner_and_script_fallback(
    tmp_path,
    monkeypatch,
):
    image_binding = ResolvedBinding(
        provider_id="fake_provider",
        provider_label="Fake Provider",
        api_style="openai_compatible",
        api_key="secret",
        base_url="https://provider.example/v1",
        timeout_seconds=300,
        model="fake-image-model",
        options={},
    )
    review_binding = ResolvedBinding(
        provider_id="review_provider",
        provider_label="Review Provider",
        api_style="openai_compatible",
        api_key="review-secret",
        base_url="https://review.example/v1",
        timeout_seconds=300,
        model="moonshotai/Kimi-K2.7-Code",
        options={},
    )
    question_binding = ResolvedBinding(
        provider_id="question_provider",
        provider_label="Question Provider",
        api_style="openai_compatible",
        api_key="question-secret",
        base_url="https://question.example/v1",
        timeout_seconds=120,
        model="deepseek-ai/DeepSeek-V4-Pro",
        options={},
    )
    captured = {}

    class FakeRegistry:
        def resolve_pipeline(self, *args, **kwargs):
            captured["require_review"] = kwargs["require_review"]
            return SimpleNamespace(
                id="fake_pipeline",
                label="Fake Pipeline",
                image=image_binding,
                review=review_binding,
                questions=question_binding,
                max_revisions=0,
            )

    class FakeLateralityPlanner:
        def __init__(self, **kwargs):
            captured["planner_model"] = kwargs["model"]
            captured["planner_api_key"] = kwargs["api_key"]
            captured["planner_base_url"] = kwargs["base_url"]

        async def plan(self, report_text):
            return LateralityPlan(
                findings=(
                    LateralityFinding(
                        finding="右肺门淋巴结炎性增生",
                        patient_side=PatientSide.RIGHT,
                        canvas_side=CanvasSide.LEFT,
                        anatomical_anchor="右肺门",
                        forbidden_canvas_side=CanvasSide.RIGHT,
                        requires_endpoint=True,
                    ),
                ),
                source="ai",
            )

    class FakeGenerationPipeline:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, request, *, gate_enabled, progress_callback=None):
            captured["laterality_plan"] = request.laterality_plan
            image = GeneratedImage(b"fake-image", "image/png", "fake_provider", "fake-model")
            return PipelineResult(
                final_image=image,
                accepted=None,
                gate_enabled=False,
                attempts=(GenerationAttempt(image=image, review=None),),
            )

    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(webapp_main, "_load_registry", lambda: FakeRegistry())
    monkeypatch.setattr(webapp_main, "OpenAILateralityPlanner", FakeLateralityPlanner)
    monkeypatch.setattr(webapp_main, "GenerationPipeline", FakeGenerationPipeline)

    response = asyncio.run(
        _execute_generation(
            GeneratePayload(
                case_id="P001",
                report_text="右肺门淋巴结炎性增生；膀胱左后壁不规则增厚。",
                gate_enabled=False,
                generate_questions=False,
            )
        )
    )

    anchors = {
        finding["anatomical_anchor"]: finding
        for finding in response["laterality_plan"]["findings"]
    }
    assert captured["require_review"] is False
    assert captured["planner_model"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert captured["planner_api_key"] == "question-secret"
    assert captured["planner_base_url"] == "https://question.example/v1"
    assert response["laterality_plan"]["source"] == "ai_plus_script"
    assert anchors["右肺门"]["canvas_side"] == "LEFT"
    assert anchors["膀胱左后壁不规则增厚"]["canvas_side"] == "RIGHT"
    assert captured["laterality_plan"].source == "ai_plus_script"


def test_review_strength_off_overrides_legacy_gate_requirement(tmp_path, monkeypatch):
    binding = ResolvedBinding(
        provider_id="fake_provider",
        provider_label="Fake Provider",
        api_style="openai_compatible",
        api_key="secret",
        base_url="https://provider.example/v1",
        timeout_seconds=300,
        model="fake-model",
        options={},
    )
    captured = {}

    class FakeRegistry:
        def resolve_pipeline(self, *args, **kwargs):
            captured["require_review"] = kwargs["require_review"]
            return SimpleNamespace(
                id="fake_pipeline",
                label="Fake Pipeline",
                image=binding,
                review=binding,
                questions=binding,
                max_revisions=0,
            )

    class FakeGenerationPipeline:
        def __init__(self, *args, **kwargs):
            captured["reviewer"] = args[1] if len(args) > 1 else None

        async def run(self, request, *, gate_enabled, progress_callback=None):
            captured["gate_enabled"] = gate_enabled
            image = GeneratedImage(b"fake-image", "image/png", "fake_provider", "fake-model")
            return PipelineResult(
                final_image=image,
                accepted=None,
                gate_enabled=False,
                attempts=(GenerationAttempt(image=image, review=None),),
            )

    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(webapp_main, "_load_registry", lambda: FakeRegistry())
    monkeypatch.setattr(webapp_main, "GenerationPipeline", FakeGenerationPipeline)

    response = asyncio.run(
        _execute_generation(
            GeneratePayload(
                case_id="P001",
                report_text="右肾见占位性病变，考虑恶性。",
                gate_enabled=True,
                review_strength=ReviewStrength.OFF,
                generate_questions=False,
            )
        )
    )

    assert captured["require_review"] is False
    assert captured["reviewer"] is None
    assert captured["gate_enabled"] is False
    assert response["review_strength"] == "OFF"


def test_execute_generation_starts_questions_before_image_pipeline(tmp_path, monkeypatch):
    marker = {"questions_started": False}
    binding = ResolvedBinding(
        provider_id="fake_provider",
        provider_label="Fake Provider",
        api_style="openai_compatible",
        api_key="secret",
        base_url="https://provider.example/v1",
        timeout_seconds=300,
        model="fake-model",
        options={},
    )

    class FakeRegistry:
        def resolve_pipeline(self, *args, **kwargs):
            return SimpleNamespace(
                id="fake_pipeline",
                label="Fake Pipeline",
                image=binding,
                review=binding,
                questions=binding,
                max_revisions=0,
            )

    class FakeQuestionService:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, report_text):
            marker["questions_started"] = True
            await asyncio.sleep(0)
            return [{"question": "病灶在哪里？", "answer": "右肾", "options": ["右肾", "左肾"]}]

    class FakeGenerationPipeline:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, request, *, gate_enabled, progress_callback=None):
            assert marker["questions_started"] is True
            image = GeneratedImage(b"fake-image", "image/png", "fake_provider", "fake-model")
            return PipelineResult(
                final_image=image,
                accepted=None,
                gate_enabled=False,
                attempts=(GenerationAttempt(image=image, review=None),),
            )

    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(webapp_main, "_load_registry", lambda: FakeRegistry())
    monkeypatch.setattr(webapp_main, "OpenAIQuestionService", FakeQuestionService)
    monkeypatch.setattr(webapp_main, "GenerationPipeline", FakeGenerationPipeline)

    response = asyncio.run(
        _execute_generation(
            GeneratePayload(
                case_id="P001",
                report_text="右肾见占位性病变，考虑恶性。",
                gate_enabled=False,
                generate_questions=True,
            )
        )
    )

    assert response["questions"][0]["answer"] == "右肾"


def test_gate_outcome_normalizes_ai_error_categories_for_statistics():
    result = PipelineResult(
        final_image=GeneratedImage(b"image", "image/png", "fake", "fake"),
        accepted=False,
        gate_enabled=True,
        attempts=(
            GenerationAttempt(
                image=GeneratedImage(b"image", "image/png", "fake", "fake"),
                review=ReviewDecision(
                    passed=False,
                    issues=(
                        ReviewIssue(category="laterality", description="左右错误"),
                        ReviewIssue(category="text", description="中文错误"),
                    ),
                ),
            ),
        ),
    )

    assert _gate_outcome(result)["finalErrorTypes"] == [
        "LATERALITY",
        "CHINESE_TEXT",
    ]


def test_gate_outcome_identifies_returned_revision_and_attempt_count():
    first = GeneratedImage(b"first", "image/png", "fake", "fake")
    revised = GeneratedImage(b"revised", "image/png", "fake", "fake")
    failure = ReviewDecision(passed=False, summary="重修后仍有侧别错误")
    result = PipelineResult(
        final_image=revised,
        accepted=False,
        gate_enabled=True,
        attempts=(
            GenerationAttempt(image=first, review=failure),
            GenerationAttempt(image=revised, review=failure),
        ),
    )

    outcome = _gate_outcome(result)

    assert outcome["attemptCount"] == 2
    assert outcome["revisionCount"] == 1
    assert outcome["returnedAttempt"] == 2


def _write_history_manifest(
    runtime_dir,
    *,
    run_id: str,
    case_id: str,
    generated_at: str,
    image_content: bytes = b"fake-image",
):
    case_dir = runtime_dir / generated_at[:10] / case_id / run_id
    case_dir.mkdir(parents=True)
    (case_dir / "attempt_1.png").write_bytes(image_content)
    manifest = {
        "run_id": run_id,
        "generated_at": generated_at,
        "duration_seconds": 12.5,
        "case_id": case_id,
        "report_text": "右肺门淋巴结炎性增生。",
        "strategy": "GATED",
        "gate_enabled": True,
        "accepted": True,
        "gate_outcome": {"status": "PASS", "reason": "通过"},
        "pipeline": {"label": "Fake Pipeline"},
        "attempts": [{"image": "attempt_1.png", "review": None}],
        "questions": [],
        "human_evaluation": None,
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return case_dir


def test_history_endpoint_lists_runs_newest_first_with_image_links(tmp_path, monkeypatch):
    old_run_id = "1" * 32
    new_run_id = "2" * 32
    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    _write_history_manifest(
        tmp_path,
        run_id=old_run_id,
        case_id="P001",
        generated_at="2026-06-21T09:30:00+00:00",
    )
    _write_history_manifest(
        tmp_path,
        run_id=new_run_id,
        case_id="P002",
        generated_at="2026-06-23T09:30:00+00:00",
    )

    response = asyncio.run(list_history(limit=10))

    assert [item["run_id"] for item in response["items"]] == [new_run_id, old_run_id]
    assert response["items"][0]["image_url"] == (
        f"/api/runs/{new_run_id}/files/attempt_1.png"
    )
    assert response["items"][0]["gate_outcome"]["status"] == "PASS"
    assert response["items"][0]["attempt_count"] == 1


def test_history_endpoint_can_filter_by_case_id(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    _write_history_manifest(
        tmp_path,
        run_id="1" * 32,
        case_id="P001",
        generated_at="2026-06-21T09:30:00+00:00",
    )
    _write_history_manifest(
        tmp_path,
        run_id="2" * 32,
        case_id="P002",
        generated_at="2026-06-23T09:30:00+00:00",
    )

    response = asyncio.run(list_history(limit=10, case_id="P001"))

    assert [item["case_id"] for item in response["items"]] == ["P001"]


def test_get_run_loads_manifest_and_embeds_final_image_data_url(tmp_path, monkeypatch):
    run_id = "3" * 32
    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    _write_history_manifest(
        tmp_path,
        run_id=run_id,
        case_id="P001",
        generated_at="2026-06-23T09:30:00+00:00",
        image_content=b"image-bytes",
    )

    response = asyncio.run(get_run(run_id))

    assert response["run_id"] == run_id
    assert response["image_url"] == f"/api/runs/{run_id}/files/attempt_1.png"
    assert response["image_data_url"] == "data:image/png;base64,aW1hZ2UtYnl0ZXM="


def test_run_file_endpoint_serves_only_safe_run_files(tmp_path, monkeypatch):
    run_id = "4" * 32
    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    _write_history_manifest(
        tmp_path,
        run_id=run_id,
        case_id="P001",
        generated_at="2026-06-23T09:30:00+00:00",
    )

    response = asyncio.run(get_run_file(run_id, "attempt_1.png"))

    assert response.media_type == "image/png"
    assert str(response.path).endswith("attempt_1.png")
    with pytest.raises(HTTPException):
        asyncio.run(get_run_file(run_id, "../manifest.json"))
    with pytest.raises(HTTPException):
        asyncio.run(get_run_file(run_id, "manifest.json"))


def test_save_evaluation_updates_run_manifest(tmp_path, monkeypatch):
    run_id = "a" * 32
    case_dir = tmp_path / "2026-06-21" / "P001" / run_id
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "human_evaluation": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)
    evaluation = HumanEvaluation(
        overallDecision=OverallDecision.FAIL,
        errorTypes=[ErrorType.LATERALITY],
        reviewer="doctor-a",
    )

    response = asyncio.run(save_evaluation(run_id, evaluation))

    saved = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    assert response["human_evaluation"]["overallDecision"] == "FAIL"
    assert saved["human_evaluation"]["errorTypes"] == ["LATERALITY"]


def test_case_directory_still_finds_legacy_flat_runs(tmp_path, monkeypatch):
    run_id = "b" * 32
    legacy_case_dir = tmp_path / run_id
    legacy_case_dir.mkdir()
    monkeypatch.setattr(webapp_main, "RUNTIME_DIR", tmp_path)

    assert webapp_main._case_directory(run_id) == legacy_case_dir


def test_upstream_404_error_identifies_stage_without_leaking_exception_text():
    binding = ResolvedBinding(
        provider_id="question_api",
        provider_label="SiliconFlow",
        api_style="openai_compatible",
        api_key="secret-key",
        base_url="https://api.siliconflow.cn/v1",
        timeout_seconds=300,
        model="deepseek-ai/DeepSeek-V4-Pro",
        options={},
    )

    class FakeNotFoundError(Exception):
        status_code = 404

    detail = _upstream_error_detail(
        "questions",
        binding,
        FakeNotFoundError("secret-key must never be shown"),
    )

    assert detail["code"] == "UPSTREAM_NOT_FOUND"
    assert detail["stage"] == "questions"
    assert detail["providerLabel"] == "SiliconFlow"
    assert detail["upstreamStatus"] == 404
    assert "secret-key" not in json.dumps(detail)


def test_generate_stream_emits_progress_then_result(monkeypatch):
    async def fake_execute(payload, progress_callback=None):
        await progress_callback(
            {
                "type": "progress",
                "stage": "configuration",
                "status": "completed",
                "message": "配置检查完成",
            }
        )
        return {"run_id": "b" * 32, "image_data_url": "data:image/png;base64,aW1n"}

    monkeypatch.setattr(webapp_main, "_execute_generation", fake_execute)
    payload = GeneratePayload(
        case_id="P001",
        report_text="右肾见占位性病变，考虑恶性。",
        gate_enabled=False,
        generate_questions=False,
    )

    async def collect_events():
        response = await generate_stream(payload)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return [json.loads(line) for line in "".join(chunks).splitlines()]

    events = asyncio.run(collect_events())

    assert events[0]["stage"] == "configuration"
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["run_id"] == "b" * 32
