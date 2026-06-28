"""Concrete OpenAI and OpenAI-compatible provider adapters."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from urllib.request import urlopen

import aiohttp
from openai import OpenAI
from pydantic import BaseModel, Field

from petct.display_plan import (
    DisplayDetailField,
    DisplayFindingType,
    DisplayPlan,
    ReviewStrength,
    normalize_display_selection,
    validate_display_plan_sources,
)
from petct.generation import (
    GeneratedImage,
    GenerationRequest,
    ReviewDecision,
)
from petct.laterality import (
    CanvasSide,
    LateralityFinding,
    LateralityPlan,
    PatientSide,
    canvas_side_for,
    forbidden_canvas_side_for,
)


DISPLAY_PLAN_PROMPT_VERSION = "petct-display-plan-2026-06-28.1"
LATERALITY_PLANNER_PROMPT_VERSION = "petct-laterality-planner-2026-06-21.1"
REVIEW_PROMPT_VERSION = "petct-review-2026-06-23.3"


class _ReviewSchema(BaseModel):
    passed: bool
    reason: str = Field(min_length=1)


class _LateralityFindingSchema(BaseModel):
    finding: str = Field(min_length=1)
    patient_side: PatientSide
    anatomical_anchor: str = Field(min_length=1)
    requires_endpoint: bool = True


class _LateralityPlanSchema(BaseModel):
    findings: list[_LateralityFindingSchema] = Field(default_factory=list)

    def to_domain(self) -> LateralityPlan:
        findings = []
        for item in self.findings:
            findings.append(
                LateralityFinding(
                    finding=item.finding,
                    patient_side=item.patient_side,
                    canvas_side=canvas_side_for(item.patient_side),
                    anatomical_anchor=item.anatomical_anchor,
                    forbidden_canvas_side=forbidden_canvas_side_for(item.patient_side),
                    requires_endpoint=item.requires_endpoint,
                )
            )
        return LateralityPlan(findings=tuple(findings), source="ai")


class OpenAIImageGenerator:
    """Generate or revise images with an OpenAI-compatible Image API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-image-2",
        base_url: str | None = None,
        size: str = "1536x1024",
        quality: str = "high",
        provider_id: str = "openai",
        timeout_seconds: float = 300,
        options: dict | None = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.api_key = api_key
        self.base_url = base_url
        self.provider_id = provider_id
        self.model = model
        self.options = {
            "size": size,
            "quality": quality,
            **(options or {}),
        }
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        request: GenerationRequest,
        *,
        previous_image: GeneratedImage | None = None,
        correction_prompt: str | None = None,
    ) -> GeneratedImage:
        if self._uses_async_task_api():
            return await self._generate_via_async_task(
                request,
                previous_image=previous_image,
                correction_prompt=correction_prompt,
            )

        result = await asyncio.to_thread(
            self._generate_sync,
            request,
            previous_image,
            correction_prompt,
        )
        image_data = result.data[0]
        if image_data.b64_json:
            content = base64.b64decode(image_data.b64_json)
        elif image_data.url:
            with urlopen(image_data.url, timeout=self.timeout_seconds) as response:
                content = response.read()
        else:
            raise RuntimeError("Image response contained neither b64_json nor URL")

        output_format = self.options.get("output_format", "png")
        return GeneratedImage(
            content=content,
            mime_type=f"image/{'jpeg' if output_format == 'jpeg' else output_format}",
            provider=self.provider_id,
            model=self.model,
        )

    def _generate_sync(
        self,
        request: GenerationRequest,
        previous_image: GeneratedImage | None,
        correction_prompt: str | None,
    ):
        prompt = request.prompt
        if correction_prompt:
            prompt = f"{prompt}\n\n{correction_prompt}"

        images = []
        if previous_image:
            previous_file = io.BytesIO(previous_image.content)
            previous_file.name = f"previous.{_image_extension(previous_image.mime_type)}"
            images.append(previous_file)
        if request.reference_image and not previous_image:
            reference_file = io.BytesIO(request.reference_image)
            reference_file.name = f"reference.{_image_extension(request.reference_mime_type)}"
            images.append(reference_file)

        if images:
            image_arg = images if len(images) > 1 else images[0]
            return self.client.images.edit(
                model=self.model,
                image=image_arg,
                prompt=prompt,
                **self._filtered_options(for_edit=True),
            )
        return self.client.images.generate(
            model=self.model,
            prompt=prompt,
            **self._filtered_options(for_edit=False),
        )

    def _filtered_options(self, for_edit: bool) -> dict:
        common = {
            "background",
            "n",
            "output_compression",
            "output_format",
            "partial_images",
            "quality",
            "response_format",
            "size",
            "stream",
            "user",
        }
        edit_only = set()
        if for_edit and self.model != "gpt-image-2":
            edit_only.add("input_fidelity")
        allowed = common | (edit_only if for_edit else {"moderation", "style"})
        return {
            key: value
            for key, value in self.options.items()
            if key in allowed and value is not None
        }

    def _uses_async_task_api(self) -> bool:
        if self.options.get("async_task") is True:
            return True
        base_url = (self.base_url or "").lower()
        return (
            self.model == "openai/gpt-image-2"
            and "shengsuanyun.com" in base_url
        )

    async def _generate_via_async_task(
        self,
        request: GenerationRequest,
        *,
        previous_image: GeneratedImage | None,
        correction_prompt: str | None,
    ) -> GeneratedImage:
        body = self._async_task_body(
            request,
            previous_image=previous_image,
            correction_prompt=correction_prompt,
        )
        # Shengsuanyun documents gpt-image-2 as an async task model:
        # POST /tasks/generations, then poll GET /tasks/generations/{task_id}.
        task_id = await self._create_async_task(body)
        payload = await self._wait_for_async_task(task_id)
        image = self._extract_async_task_image(payload)
        mime_type = self._mime_type_from_options()
        if image.get("b64_json"):
            content = base64.b64decode(str(image["b64_json"]))
        elif image.get("url"):
            content = await self._download_image_url(str(image["url"]))
        else:
            raise RuntimeError("Async image task completed without image data")
        return GeneratedImage(
            content=content,
            mime_type=mime_type,
            provider=self.provider_id,
            model=self.model,
        )

    def _async_task_body(
        self,
        request: GenerationRequest,
        *,
        previous_image: GeneratedImage | None,
        correction_prompt: str | None,
    ) -> dict[str, object]:
        prompt = request.prompt
        if correction_prompt:
            prompt = f"{prompt}\n\n{correction_prompt}"

        body: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
        }
        for key, value in self.options.items():
            if key in {
                "size",
                "quality",
                "output_format",
                "background",
                "moderation",
                "output_compression",
                "n",
                "seed",
                "user",
                "watermark",
                "response_format",
            } and value is not None:
                body[key] = value

        if "n" not in body:
            body["n"] = 1

        input_image = previous_image
        input_bytes = None
        input_mime = "image/png"
        if input_image:
            input_bytes = input_image.content
            input_mime = input_image.mime_type
        elif request.reference_image:
            input_bytes = request.reference_image
            input_mime = request.reference_mime_type
        if input_bytes:
            body["image"] = _image_data_url(input_bytes, input_mime)
        return body

    def _async_task_url(self, task_id: str | None = None) -> str:
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        path = "tasks/generations"
        if task_id:
            path = f"{path}/{task_id}"
        return f"{base_url}/{path}"

    async def _create_async_task(self, body: dict[str, object]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._async_task_url(),
                headers=headers,
                json=body,
            ) as response:
                payload = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(
                        f"async image task create failed: HTTP {response.status}: "
                        f"{str(payload)[:500]}"
                    )
        task_id = self._extract_async_task_id(payload)
        if not task_id:
            raise RuntimeError(f"async image task response lacks task_id: {payload}")
        return task_id

    async def _wait_for_async_task(self, task_id: str) -> dict:
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        poll_interval = float(self.options.get("async_poll_interval", 3))
        last_payload: dict | None = None
        while asyncio.get_running_loop().time() < deadline:
            payload = await self._poll_async_task(task_id)
            last_payload = payload
            status = self._extract_async_task_status(payload)
            if status in {"COMPLETED", "SUCCEEDED", "SUCCESS", "succeeded", "completed"}:
                return payload
            if status in {"FAILED", "CANCELLED", "failed", "cancelled", "error"}:
                reason = self._extract_async_task_failure_reason(payload)
                raise RuntimeError(f"async image task {status}: {reason}")
            await asyncio.sleep(poll_interval)
        raise RuntimeError(
            f"async image task timed out after {self.timeout_seconds}s: {last_payload}"
        )

    async def _poll_async_task(self, task_id: str) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = aiohttp.ClientTimeout(total=min(self.timeout_seconds, 60))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                self._async_task_url(task_id),
                headers=headers,
            ) as response:
                payload = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(
                        f"async image task poll failed: HTTP {response.status}: "
                        f"{str(payload)[:500]}"
                    )
                return payload

    async def _download_image_url(self, url: str) -> bytes:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise RuntimeError(
                        f"async image download failed: HTTP {response.status}: "
                        f"{text[:300]}"
                    )
                return await response.read()

    def _extract_async_task_id(self, payload: dict) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        candidates = []
        if isinstance(data, dict):
            candidates.extend([data.get("task_id"), data.get("id")])
        candidates.extend([payload.get("task_id"), payload.get("id")])
        return next((str(value) for value in candidates if value), "")

    def _extract_async_task_status(self, payload: dict) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict) and data.get("status"):
            return str(data["status"])
        return str(payload.get("status", ""))

    def _extract_async_task_failure_reason(self, payload: dict) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            for key in ("fail_reason", "message", "error"):
                if data.get(key):
                    return str(data[key])
        for key in ("fail_reason", "message", "error"):
            if payload.get(key):
                return str(payload[key])
        return "unknown failure"

    def _extract_async_task_image(self, payload: dict) -> dict[str, str]:
        data = payload.get("data") if isinstance(payload, dict) else None
        nested = data.get("data") if isinstance(data, dict) else None
        candidates = [nested, data, payload.get("output"), payload]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("image_urls", "images", "urls"):
                values = candidate.get(key)
                if isinstance(values, list) and values:
                    first = values[0]
                    if isinstance(first, str):
                        return {"url": first}
                    if isinstance(first, dict):
                        url = first.get("url")
                        b64_json = first.get("b64_json")
                        if url or b64_json:
                            return {
                                "url": str(url) if url else "",
                                "b64_json": str(b64_json) if b64_json else "",
                            }
            url = candidate.get("url")
            if isinstance(url, str) and url:
                return {"url": url}
            b64_json = candidate.get("b64_json")
            if isinstance(b64_json, str) and b64_json:
                return {"b64_json": b64_json}
        raise RuntimeError(f"async image task result lacks image URL: {payload}")

    def _mime_type_from_options(self) -> str:
        output_format = self.options.get("output_format", "png")
        if output_format == "jpeg":
            return "image/jpeg"
        if output_format == "webp":
            return "image/webp"
        return "image/png"


class OpenAIDisplayPlanner:
    """Filter report content into the exact findings/details to draw."""

    INSTRUCTIONS = """你是 PET-CT 患者友好图示的信息筛选员。任务是在生图和左右规划之前，把“检查结论”和可选“检查所见”合并筛选为严格的 display_plan JSON。

原则：
1. 检查结论是必填事实来源，但不代表其中每一句都必须画出；检查所见只用于补充结论相关发现的 SUVmax、尺寸、FDG 摄取和解剖细节。
2. 只能保留用户本次选择的发现类型和细节字段。未选择的良性、阴性、治疗背景、SUVmax、尺寸或 FDG 信息不得写入可绘制项目。
3. 不得新增、改写、推断报告没有的诊断、侧别、器官、病灶、SUVmax、尺寸或 FDG 描述；SUVmax 和尺寸必须逐字来自输入文本。
4. 如果结论里有多余背景或低价值信息，应放入 excludedItems 或忽略，不要为了完整而画入图中。
5. labelText 必须是患者可理解的简短中文标签；仍要保留关键侧别、部位、性质，并只加入被选择的细节。
6. contentTypes 只能使用 MALIGNANT、BENIGN、INDETERMINATE、IMPORTANT_NEGATIVE、TREATMENT_CONTEXT。
7. patientSide 只能使用 LEFT、RIGHT、BILATERAL、MIDLINE、UNSPECIFIED；这是医学患者侧别，不是画面左右。
8. 每个 draw=true 的项目必须提供 conclusionEvidence 或 findingsEvidence，尽量摘录最短原文片段。

输出：
- 只输出一个 JSON 对象，不要 Markdown、代码块、分析过程、<think> 或 JSON 外文本。
- JSON 顶层字段必须包含 selectedFindingTypes、selectedDetailFields、items、excludedItems、warnings。
- 如果所选类型下没有可绘制项目，items 可为空，但仍需在 warnings 中说明原因。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 300,
        api_style: str = "openai",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model
        self.api_style = api_style

    async def plan(
        self,
        *,
        conclusion_text: str,
        findings_text: str | None = None,
        finding_types: list[DisplayFindingType | str] | None = None,
        detail_fields: list[DisplayDetailField | str] | None = None,
    ) -> DisplayPlan:
        normalized_finding_types, normalized_detail_fields = normalize_display_selection(
            finding_types,
            detail_fields,
        )
        prompt = _display_plan_user_prompt(
            conclusion_text=conclusion_text,
            findings_text=findings_text,
            finding_types=normalized_finding_types,
            detail_fields=normalized_detail_fields,
        )
        parsed = await asyncio.to_thread(self._plan_sync, prompt)
        if parsed is None:
            raise RuntimeError("Display planner returned no structured output")
        parsed = parsed.model_copy(
            update={
                "selected_finding_types": list(normalized_finding_types),
                "selected_detail_fields": list(normalized_detail_fields),
            }
        )
        validate_display_plan_sources(
            parsed,
            conclusion_text=conclusion_text,
            findings_text=findings_text,
        )
        return parsed

    def _plan_sync(self, prompt: str) -> DisplayPlan | None:
        if self.api_style == "openai_compatible":
            schema = json.dumps(DisplayPlan.model_json_schema(), ensure_ascii=False)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{self.INSTRUCTIONS}\n"
                            f"Prompt version: {DISPLAY_PLAN_PROMPT_VERSION}.\n"
                            "只输出一个 JSON 对象，不要使用 Markdown 代码块。"
                            f"输出必须符合以下 JSON Schema：{schema}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                return None
            return DisplayPlan.model_validate_json(_extract_json_object(content))

        response = self.client.responses.parse(
            model=self.model,
            instructions=(
                f"{self.INSTRUCTIONS}\n"
                f"Prompt version: {DISPLAY_PLAN_PROMPT_VERSION}."
            ),
            input=prompt,
            text_format=DisplayPlan,
        )
        return response.output_parsed


class OpenAILateralityPlanner:
    """Convert a PET-CT report into a structured laterality plan."""

    INSTRUCTIONS = """你是 PET-CT 报告结构化抽取员。任务是在生图前把报告中的侧别和解剖定位转换成稳定 JSON，供后续图片生成和审查使用。

只使用报告原文，不得新增发现、诊断、侧别、器官或数值。不要输出解释、Markdown 或额外字段。

抽取范围：
1. 抽取所有包含患者左侧、右侧、双侧/两侧的发现。
2. 抽取中线定位但对示意图有重要定位价值的发现，例如前列腺、膀胱中线、纵隔、隆突下、脊柱/椎体。
3. 一个长句包含多条发现时必须拆成多条 findings，不要把“右肺门淋巴结”和“膀胱左后壁”合并到同一条。
4. 既往史、术后、化疗后、治疗后等背景信息可以保留，但 requires_endpoint 应为 false，除非该短句同时描述了当前病灶、结节、增厚、代谢异常、淋巴结、囊肿、炎症、积液或转移。

字段规则：
- finding：报告原文中的最短独立发现短句，保留侧别和关键限定语。
- patient_side：只能是 LEFT、RIGHT、BILATERAL、MIDLINE 或 UNSPECIFIED。医学患者右侧为 RIGHT，患者左侧为 LEFT；双侧/两侧为 BILATERAL；前列腺、纵隔、隆突下、椎体等中线项为 MIDLINE。
- anatomical_anchor：最短解剖锚点，例如“右肺门”“甲状腺右叶”“膀胱左后壁”“两侧睾丸”“T11椎体”。
- requires_endpoint：如果图片上必须能核对病灶点、器官局部或引导线终点，则为 true；单纯既往史/术后背景为 false。

侧别只按医学报告判断，不要转换成画面左右；画面左右由程序计算。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 300,
        api_style: str = "openai",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model
        self.api_style = api_style

    async def plan(self, report_text: str) -> LateralityPlan:
        parsed = await asyncio.to_thread(self._plan_sync, report_text)
        if parsed is None:
            raise RuntimeError("Laterality planner returned no structured output")
        return parsed.to_domain()

    def _plan_sync(self, report_text: str) -> _LateralityPlanSchema | None:
        if self.api_style == "openai_compatible":
            schema = json.dumps(_LateralityPlanSchema.model_json_schema(), ensure_ascii=False)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{self.INSTRUCTIONS}\n"
                            "只输出一个 JSON 对象，不要使用 Markdown 代码块。"
                            f"输出必须符合以下 JSON Schema：{schema}"
                        ),
                    },
                    {"role": "user", "content": report_text},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                return None
            return _LateralityPlanSchema.model_validate_json(_extract_json_object(content))

        response = self.client.responses.parse(
            model=self.model,
            instructions=self.INSTRUCTIONS,
            input=report_text,
            text_format=_LateralityPlanSchema,
        )
        return response.output_parsed


class OpenAIVisualReviewer:
    """Review a generated image against its source report and optional template."""

    LATERALITY_REVIEW_INSTRUCTIONS = """你是 PET-CT 患者友好可视化图片的专门左右审查员。
本轮只审查结构化左右清单中的侧别、器官/区域和引导线终点，不评价风格美观。门诊场景以快速拦截重大错误为目标，不做精细解剖评分。

审查输入包含原始报告、结构化左右清单和待审查图片。固定正面视角：患者右侧位于画面左侧；患者左侧位于画面右侧；双侧发现应能让患者看出涉及两侧或中线/成组区域；中线项不得被错误放到单侧。

逐项审查规则：
1. 只对 requires_endpoint=true 的条目强制寻找病灶点、器官局部或引导线终点。requires_endpoint=false 的既往史或上下文不要强制要求病灶点，但若图片主动画出，也必须符合侧别。
2. 实际侧别只看器官、病灶点或引导线终点相对人体中线的位置。标签排在哪一侧不能作为通过依据。
3. 每个左右条目都要核对：报告患者侧别、预期画面侧、实际引导线终点画面侧、实际器官/区域。
4. 只有 requires_endpoint=true 的关键条目明确缺失、方向明显反转、实际落点在 forbidden_canvas_side、落到对侧器官、或只能从标签文字猜测位置且会影响患者理解时，才判为不通过。
5. 判为不通过前，必须先在结构化左右清单中找到对应条目，并在 reason 中逐字写出该条目的 anatomical_anchor；不能只写“患者右侧某病灶”或“患者左侧某病灶”。
6. 如果病灶点/引导线终点已经落在预期画面侧，即使标签文字排在另一侧，也应按该条目通过；不得把标签排版侧误当成病灶侧。
7. 双侧条目只在确实缺少一侧落点、或两侧落点明显不对称且影响理解时判不通过；若两侧器官上均有落点，或用一个总标签清楚覆盖双侧区域，不得仅因一条标签线从单侧引出而判失败。
8. 涉及左右错误的 reason 必须写成：“【anatomical_anchor】报告为患者X侧；正面图中应位于画面X侧；当前病灶点/引导线终点实际位于画面X侧；请仅移动错误落点至画面X侧并保留正确内容”。

输出要求：
- 只输出 passed 和 reason 两个字段，不要输出 Markdown 或额外说明。
- passed=true 时，reason 写“通过：结构化左右清单中的需要落点条目均符合患者侧别与画面侧别映射”。
- passed=false 时，reason 用 1-3 句列出最关键左右错误；每句必须包含一个结构化清单中的 anatomical_anchor 原文，并直接写给下一轮 image edit 使用。"""

    REVIEW_INSTRUCTIONS = """你是 PET-CT 患者友好可视化图片的质量门控审查员。
审查输入包含两类材料：原始报告文本和待审查图片。必须同时使用两者交叉核对，不得只看图片的视觉效果，也不得只根据报告想象图片内容。
报告原文是唯一医学事实来源。先逐项重建报告中的关键发现清单，再把清单与图片中的病灶点、引导线、标签文字、颜色和解剖位置对照。
门诊场景强调生成总耗时和可用性：本门控用于拦截会明显误导患者或影响临床解释的重大错误，不用于追求每个亚区、每条标签线都完美。
如果无法清楚读取图片内容或模型无法实际查看图片，必须判定不通过并说明原因，不得臆测通过。

审查流程：
1. 先从报告中内部列出所有需要核对的发现，包括阴性或低风险但被图中展示的发现。
2. 再逐项检查图片中是否有唯一、正确的视觉对应：病灶点、器官/区域、侧别、标签、引导线、颜色、FDG/SUVmax。
3. 最后检查图片是否新增了报告没有的器官异常、病灶、数值或结论。
4. 只要报告中的关键发现均被正确表达，且不存在新增严重医学事实或明显误导患者的问题，即可通过；不因轻微排版、合理合并、非关键良性发现未画成独立落点而失败。

左右侧别强制审查算法：
1. 第一优先级审查侧别。第一步先确定人体中线和统一正面朝向；固定映射是患者左侧对应画面右侧、患者右侧对应画面左侧。若头部、躯干、骨盆朝向不一致或正面朝向不清，判定不通过并说明“方向不清”，不得继续猜测左右。
2. 从报告中提取每个“左、右、双侧”发现，逐项建立内部侧别核对表，至少包含：报告侧别、预期画面侧、实际落点画面侧、对应器官/区域、结论。
3. 实际侧别只依据器官、病灶点或引导线端点相对人体中线的落点。标签位于画面哪边不能作为侧别证据；标签在左侧排版但引导线指向画面右侧，实际定位仍是画面右侧。
4. 使用成对器官校准：患者右肺/右肾/右乳腺在画面左侧，患者左肺/左肾/左乳腺在画面右侧；肺叶、甲状腺叶、输尿管、肾上腺、髂血管旁区域和四肢同样处理。
5. 患者左侧病灶实际落在画面右侧时，不得误判为左右错误；患者右侧病灶实际落在画面左侧时，也不得误判。只有实际落点违反固定映射时才报告左右错误。
6. 文字和落点必须分别核对。文字侧别正确但点位在对侧仍不通过；点位正确但文字写成对侧也不通过。不得通过改写文字来掩盖点位错误。
7. 涉及左右错误的理由必须使用无歧义格式：“报告为患者X侧；正面图中应位于画面X侧；当前病灶点/引导线端点实际位于画面X侧；请仅移动错误落点至画面X侧并保留正确内容”。

逐项核对：

1. 完整性与幻觉：报告中的关键病灶是否遗漏；图片是否新增报告未提及的病灶、数值、器官异常或结论。
2. 解剖准确性：器官形态、相对位置、层次关系和病灶解剖分区是否合理；淋巴结是否位于腹膜后、髂血管旁、肠系膜等报告所述区域，而不是画入胃、肾、肠腔或无关器官；是否为了完整而绘制了过多无关器官。
3. 左右侧别：严格执行上方强制审查算法，逐项核对所有“左、右、双侧”发现。成图不应出现 R/L 或镜像规则说明。
4. 性质与颜色：恶性/转移/疑似转移、良性/炎症和性质未定发现的颜色是否服从报告；不得仅凭 FDG 增高或 SUVmax 高低推断恶性；每个病灶圆点、引导线端点和标签文字颜色是否与同一项发现一致。
5. FDG 与 SUVmax：数值、定性描述、病灶对应关系是否准确；不得遗漏、改写、估算或错配。
6. 中文标注：是否完整、可辨认、无乱码错字；精简后是否仍保持原意；引导线是否指向正确结构。
7. 患者友好与风格：是否为简洁温和的二维信息图，而非写实影像、完整解剖图谱或过度专业化界面；如提供参考图，只比较版式和视觉风格，不得复制其医学内容。

患者友好粒度：
- 对“左侧额叶及颞叶”“腹膜后及髂血管旁”等报告合并描述的相邻区域，图中一个清楚覆盖对应大区且侧别正确的落点/引导线可以接受；不要因为未拆成多个细分落点而判不通过。
- 对“前列腺前部及两侧外周带”“两肺炎症”“全身多发骨转移”等复合、双侧或多发发现，允许用一个总标签加成组落点、左右肺内各自落点、或代表性多发骨点表达；不要因为没有每个亚区单独引导线而判不通过。
- 只有当落点位于错误侧别、错误器官大区，或遗漏会让患者理解方向明显错误时，才作为关键定位错误。

错误分类：
- laterality：左右侧错误。
- text：中文乱码、错字、残缺或标签指代错误。
- lesion_location：病灶解剖分区或引导线定位错误。
- omission：报告中的关键病灶或关键信息未呈现。
- anatomical_distortion：器官形态、位置或层次关系明显不合理。
- benign_malignant：性质倾向或颜色编码错误。
- suvmax：SUVmax/FDG 描述遗漏、数值错误或病灶错配。
- style：不够患者友好、无关器官过多或与参考风格明显不一致。
- hallucination：新增报告中不存在的医学事实。
- other：以上类别无法覆盖的问题。

一票否决：
- 明确左右反转、关键病灶遗漏、危险幻觉、明显解剖错误、性质颜色错误或关键 SUVmax 错误等重大错误不得通过。
- 图片中文字无法辨认、关键标签无法对应到引导线、或模型无法确认图片内容时，不得通过。
- 如果只是复合区域粒度、标签线数量、代表性落点选择等不影响患者理解的细节疑虑，应判 PASS；不要为了追求细节完美触发耗时重修。

输出要求：
- 只输出 passed 和 reason 两个字段，不要输出错误分类、分数、Markdown 或额外说明。
- passed 只能是 true 或 false；reason 必须写明“报告依据 + 图片问题 + 修图要求”。
- 涉及左右错误时，reason 必须同时写清患者侧别及其应在的画面侧别，例如“患者左侧应位于画面右侧”，避免下一轮 edit 再次镜像。
- 如果 passed=true，reason 用一句话说明“通过：图片与报告关键发现一致”。
- 如果 passed=false，reason 用 1-3 句列出最关键的不通过原因，并直接写给下一轮 image edit 使用，例如“报告写右髂血管旁及右下腹肠系膜淋巴结代谢增高，但图片将淋巴结画成腹膜后单一区域；请在上一张图基础上把对应淋巴结移到正确区域，并保留其他无误内容”。"""

    STRICT_REVIEW_APPENDIX = """\n\n严格档附加规则：
- 逐项核对 display_plan 与结构化左右清单，不允许遗漏任何 draw=true 项目。
- 对 SUVmax、尺寸、FDG 摄取和良恶性颜色采用一票否决；只要数值改写、错配或遗漏，就判不通过。
- 对标签文字错字、乱码、侧别文字与落点不一致、引导线终点不明确，均判不通过。
- 不因“整体看起来合理”而忽略局部医学事实错误。"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4",
        base_url: str | None = None,
        timeout_seconds: float = 300,
        api_style: str = "openai",
        review_strength: ReviewStrength | str = ReviewStrength.STANDARD,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model
        self.api_style = api_style
        self.review_strength = ReviewStrength(review_strength)

    async def review(
        self,
        request: GenerationRequest,
        image: GeneratedImage,
    ) -> ReviewDecision:
        if (
            self.review_strength != ReviewStrength.QUICK
            and _should_run_laterality_precheck(request)
        ):
            parsed = await asyncio.to_thread(
                self._review_sync,
                request,
                image,
                self.LATERALITY_REVIEW_INSTRUCTIONS,
            )
            if parsed is None:
                raise RuntimeError("Visual reviewer returned no structured output")
            if not parsed.passed:
                if not _reason_mentions_laterality_anchor(
                    parsed.reason,
                    request.laterality_plan,
                ):
                    parsed = None
                else:
                    return ReviewDecision(
                        passed=False,
                        issues=(),
                        summary=parsed.reason,
                    )
            if parsed is not None and not parsed.passed:
                return ReviewDecision(
                    passed=False,
                    issues=(),
                    summary=parsed.reason,
                )

        parsed = await asyncio.to_thread(
            self._review_sync,
            request,
            image,
            self._comprehensive_instructions(),
        )
        if parsed is None:
            raise RuntimeError("Visual reviewer returned no structured output")
        return ReviewDecision(
            passed=parsed.passed,
            issues=(),
            summary=parsed.reason,
        )

    @classmethod
    def instruction_bundle(cls) -> str:
        return (
            f"{cls.LATERALITY_REVIEW_INSTRUCTIONS}\n\n"
            f"{cls.REVIEW_INSTRUCTIONS}\n\n"
            f"{cls.STRICT_REVIEW_APPENDIX}"
        )

    def _comprehensive_instructions(self) -> str:
        if self.review_strength == ReviewStrength.STRICT:
            return f"{self.REVIEW_INSTRUCTIONS}{self.STRICT_REVIEW_APPENDIX}"
        return self.REVIEW_INSTRUCTIONS

    def _review_sync(
        self,
        request: GenerationRequest,
        image: GeneratedImage,
        instructions: str,
    ):
        if self.api_style == "openai_compatible":
            return self._review_compatible_sync(request, image, instructions)

        review_context = _review_context_text(request)
        content = [
            {
                "type": "input_text",
                "text": review_context,
            },
            {
                "type": "input_image",
                "image_url": _image_data_url(image.content, image.mime_type),
                "detail": "high",
            },
        ]
        if request.reference_image:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": "下面第二张图片是目标样式模板，仅用于比较版式与视觉风格。",
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(
                            request.reference_image,
                            request.reference_mime_type,
                        ),
                        "detail": "high",
                    },
                ]
            )
        response = self.client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text_format=_ReviewSchema,
        )
        return response.output_parsed

    def _review_compatible_sync(
        self,
        request: GenerationRequest,
        image: GeneratedImage,
        instructions: str,
    ) -> _ReviewSchema | None:
        review_context = _review_context_text(request)
        content = [
            {
                "type": "text",
                "text": review_context,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(image.content, image.mime_type),
                    "detail": "high",
                },
            },
        ]
        if request.reference_image:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "下面第二张图片是目标样式模板，仅用于比较版式与视觉风格。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(
                                request.reference_image,
                                request.reference_mime_type,
                            ),
                            "detail": "high",
                        },
                    },
                ]
            )
        schema = json.dumps(_ReviewSchema.model_json_schema(), ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                            f"{instructions}\n"
                            "只输出一个 JSON 对象，不要使用 Markdown 代码块、<think>、"
                            "分析过程或任何 JSON 外文本。"
                            f"输出必须符合以下 JSON Schema：{schema}"
                        ),
                },
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        response_content = response.choices[0].message.content
        if not response_content:
            return None
        return _ReviewSchema.model_validate_json(_extract_json_object(response_content))


def _image_data_url(content: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _should_run_laterality_precheck(request: GenerationRequest) -> bool:
    if not request.laterality_plan:
        return False
    sided_values = {PatientSide.LEFT, PatientSide.RIGHT, PatientSide.BILATERAL}
    return any(
        finding.requires_endpoint and finding.patient_side in sided_values
        for finding in request.laterality_plan.findings
    )


def _reason_mentions_laterality_anchor(
    reason: str,
    laterality_plan: LateralityPlan | None,
) -> bool:
    if not laterality_plan:
        return False
    normalized_reason = reason.replace(" ", "")
    for finding in laterality_plan.findings:
        if not finding.requires_endpoint:
            continue
        anchor = finding.anatomical_anchor.replace(" ", "")
        if anchor and anchor in normalized_reason:
            return True
    return False


def _review_context_text(request: GenerationRequest) -> str:
    display_block = (
        f"\n\n{request.display_plan.to_prompt_block()}"
        if request.display_plan
        else ""
    )
    laterality_block = (
        f"\n\n{request.laterality_plan.to_review_block()}"
        if request.laterality_plan
        else ""
    )
    return (
        f"病例ID：{request.case_id}\n"
        f"原始报告：\n{request.report_text}"
        f"{display_block}"
        f"{laterality_block}\n\n"
        "请同时根据上方原始报告、展示计划、结构化左右清单和下面第一张图片进行交叉审查。"
    )


def _display_plan_user_prompt(
    *,
    conclusion_text: str,
    findings_text: str | None,
    finding_types: tuple[DisplayFindingType, ...],
    detail_fields: tuple[DisplayDetailField, ...],
) -> str:
    selected_findings = ", ".join(item.value for item in finding_types)
    selected_details = (
        ", ".join(item.value for item in detail_fields)
        if detail_fields
        else "无"
    )
    findings_source = findings_text.strip() if findings_text and findings_text.strip() else "（未提供）"
    return (
        "请生成 display_plan。只允许展示本次选择的内容，并把未采用但有意义的原文写入 excludedItems。\n\n"
        f"本次需要展示的发现类型：{selected_findings}\n"
        f"本次允许加入的细节字段：{selected_details}\n\n"
        "字段含义：\n"
        "- MALIGNANT：恶性、转移、考虑恶性或考虑转移。\n"
        "- BENIGN：明确良性或炎症/术后改变。\n"
        "- INDETERMINATE：性质未定、建议随访或不能排除。\n"
        "- IMPORTANT_NEGATIVE：对患者理解重要的阴性结论。\n"
        "- TREATMENT_CONTEXT：既往治疗、术后、化疗后等背景。\n"
        "- SUVMAX：只保留原文 SUVmax 数值。\n"
        "- FDG_UPTAKE：只保留原文 FDG/代谢增高或减低描述。\n"
        "- LESION_SIZE：只保留原文尺寸。\n"
        "- ANATOMICAL_DETAIL：保留必要解剖分区。\n\n"
        "检查结论：\n"
        f"{conclusion_text.strip()}\n\n"
        "检查所见：\n"
        f"{findings_source}"
    )


def _image_extension(mime_type: str) -> str:
    return {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
        mime_type,
        "png",
    )


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _extract_json_object(content: str) -> str:
    text = _strip_json_fence(content)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return text[index : index + end]
    return text
