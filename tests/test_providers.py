import asyncio
import json
from types import SimpleNamespace

from petct.generation import GeneratedImage, GenerationRequest
from petct.display_plan import DisplayDetailField, DisplayFindingType
from petct.laterality import CanvasSide, PatientSide, build_laterality_plan
from petct.providers import (
    OpenAIImageGenerator,
    OpenAIDisplayPlanner,
    OpenAILateralityPlanner,
    OpenAIVisualReviewer,
    _image_data_url,
)


def test_image_data_url_contains_mime_type_and_base64():
    assert _image_data_url(b"png", "image/png") == "data:image/png;base64,cG5n"


def test_shengsuanyun_image2_generation_uses_async_task_api():
    created_bodies = []
    polled_task_ids = []
    downloaded_urls = []
    generator = OpenAIImageGenerator(
        api_key="test-key",
        model="openai/gpt-image-2",
        base_url="https://router.shengsuanyun.com/api/v1",
        provider_id="shengsuanyun",
        options={"size": "1536x1024", "quality": "high", "output_format": "png"},
    )

    async def fake_create(body):
        created_bodies.append(body)
        return "task_123"

    async def fake_poll(task_id):
        polled_task_ids.append(task_id)
        return {
            "code": "success",
            "data": {
                "task_id": task_id,
                "status": "COMPLETED",
                "data": {"image_urls": ["https://cdn.example/image.png"]},
            },
        }

    async def fake_download(url):
        downloaded_urls.append(url)
        return b"async-png"

    generator._create_async_task = fake_create
    generator._poll_async_task = fake_poll
    generator._download_image_url = fake_download

    result = asyncio.run(
        generator.generate(GenerationRequest("P001", "右肾病灶", "生成医学图解"))
    )

    assert result.content == b"async-png"
    assert result.mime_type == "image/png"
    assert created_bodies == [
        {
            "model": "openai/gpt-image-2",
            "prompt": "生成医学图解",
            "size": "1536x1024",
            "quality": "high",
            "output_format": "png",
            "n": 1,
        }
    ]
    assert polled_task_ids == ["task_123"]
    assert downloaded_urls == ["https://cdn.example/image.png"]
    assert generator._async_task_url() == (
        "https://router.shengsuanyun.com/api/v1/tasks/generations"
    )


def test_shengsuanyun_image2_revision_sends_previous_image_to_async_task():
    created_bodies = []
    generator = OpenAIImageGenerator(
        api_key="test-key",
        model="openai/gpt-image-2",
        base_url="https://router.shengsuanyun.com/api/v1",
        provider_id="shengsuanyun",
        options={"size": "1024x1024", "quality": "medium"},
    )

    async def fake_create(body):
        created_bodies.append(body)
        return "task_456"

    async def fake_poll(task_id):
        return {
            "data": {
                "task_id": task_id,
                "status": "COMPLETED",
                "data": {"image_urls": ["https://cdn.example/revised.png"]},
            },
        }

    async def fake_download(url):
        return b"revised-png"

    generator._create_async_task = fake_create
    generator._poll_async_task = fake_poll
    generator._download_image_url = fake_download

    result = asyncio.run(
        generator.generate(
            GenerationRequest("P002", "右肺门淋巴结", "原始提示词"),
            previous_image=GeneratedImage(b"old-image", "image/png", "fake", "fake"),
            correction_prompt="只修复右肺门落点。",
        )
    )

    assert result.content == b"revised-png"
    assert len(created_bodies) == 1
    assert created_bodies[0]["model"] == "openai/gpt-image-2"
    assert created_bodies[0]["prompt"].endswith("只修复右肺门落点。")
    assert created_bodies[0]["image"] == _image_data_url(b"old-image", "image/png")


def test_compatible_laterality_planner_returns_program_mapped_canvas_sides():
    calls = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "<think>先分析报告</think>\n"
                                + json.dumps(
                                    {
                                        "findings": [
                                            {
                                                "finding": "右肺门淋巴结炎性增生",
                                                "patient_side": "RIGHT",
                                                "anatomical_anchor": "右肺门",
                                                "requires_endpoint": True,
                                            },
                                            {
                                                "finding": "膀胱左后壁不规则增厚",
                                                "patient_side": "LEFT",
                                                "anatomical_anchor": "膀胱左后壁",
                                                "requires_endpoint": True,
                                            },
                                        ]
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        )
                    )
                ]
            )

    planner = OpenAILateralityPlanner(
        api_key="test-key",
        model="text-model",
        base_url="https://gateway.example/v1",
        api_style="openai_compatible",
    )
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions()))

    plan = asyncio.run(
        planner.plan("右肺门淋巴结炎性增生；膀胱左后壁不规则增厚。")
    )

    assert plan.source == "ai"
    assert plan.findings[0].patient_side == PatientSide.RIGHT
    assert plan.findings[0].canvas_side == CanvasSide.LEFT
    assert plan.findings[0].forbidden_canvas_side == CanvasSide.RIGHT
    assert plan.findings[1].patient_side == PatientSide.LEFT
    assert plan.findings[1].canvas_side == CanvasSide.RIGHT
    assert "JSON" in calls[0]["messages"][0]["content"]


def test_compatible_display_planner_returns_validated_display_plan():
    calls = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "<think>筛选图中展示内容</think>\n"
                                + json.dumps(
                                    {
                                        "selectedFindingTypes": ["MALIGNANT"],
                                        "selectedDetailFields": ["SUVMAX", "LESION_SIZE"],
                                        "items": [
                                            {
                                                "id": "item-1",
                                                "contentTypes": ["MALIGNANT"],
                                                "priority": "primary",
                                                "anatomy": "右肾",
                                                "patientSide": "RIGHT",
                                                "nature": "考虑恶性",
                                                "colorClass": "malignant_suspected",
                                                "labelText": "右肾占位，考虑恶性，SUVmax 12.9，约3.2 cm",
                                                "suvmax": "12.9",
                                                "size": "3.2 cm",
                                                "conclusionEvidence": "右肾占位，考虑恶性",
                                                "findingsEvidence": "右肾见软组织肿块，约3.2 cm，SUVmax=12.9",
                                                "confidence": "high",
                                            }
                                        ],
                                        "excludedItems": [],
                                        "warnings": [],
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        )
                    )
                ]
            )

    planner = OpenAIDisplayPlanner(
        api_key="test-key",
        model="text-model",
        base_url="https://gateway.example/v1",
        api_style="openai_compatible",
    )
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions()))

    plan = asyncio.run(
        planner.plan(
            conclusion_text="右肾占位，考虑恶性。",
            findings_text="右肾见软组织肿块，约3.2 cm，SUVmax=12.9。",
            finding_types=[DisplayFindingType.MALIGNANT],
            detail_fields=[DisplayDetailField.SUVMAX, DisplayDetailField.LESION_SIZE],
        )
    )

    assert plan.items[0].anatomy == "右肾"
    assert plan.items[0].suvmax == "12.9"
    system_prompt = calls[0]["messages"][0]["content"]
    user_prompt = calls[0]["messages"][1]["content"]
    assert "display_plan" in system_prompt
    assert "MALIGNANT" in user_prompt
    assert "检查所见" in user_prompt


def test_compatible_visual_reviewer_extracts_json_after_thinking_text():
    class FakeChatCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "<think>先核对图片和报告</think>\n"
                                '{"passed": true, "reason": "通过：图片与报告关键发现一致"}'
                            )
                        )
                    )
                ]
            )

    reviewer = OpenAIVisualReviewer(
        api_key="test-key",
        model="compatible-vision-model",
        base_url="https://gateway.example/v1",
        api_style="openai_compatible",
    )
    reviewer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions())
    )

    decision = asyncio.run(
        reviewer.review(
            GenerationRequest("P001", "右肾病灶", "生成图片"),
            GeneratedImage(b"image", "image/png", "fake", "fake"),
        )
    )

    assert decision.passed is True
    assert decision.summary == "通过：图片与报告关键发现一致"


def test_compatible_visual_reviewer_uses_chat_completions():
    calls = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {"passed": True, "reason": "通过"},
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            )

    reviewer = OpenAIVisualReviewer(
        api_key="test-key",
        model="compatible-vision-model",
        base_url="https://gateway.example/v1",
        api_style="openai_compatible",
    )
    reviewer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions()),
        responses=SimpleNamespace(
            parse=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("Responses API must not be used")
            )
        ),
    )

    decision = asyncio.run(
        reviewer.review(
            GenerationRequest(
                "P001",
                "右肾病灶",
                "生成图片",
                laterality_plan=build_laterality_plan("右肾病灶"),
            ),
            GeneratedImage(b"image", "image/png", "fake", "fake"),
        )
    )

    assert decision.passed is True
    assert len(calls) == 2
    assert "专门左右审查" in calls[0]["messages"][0]["content"]
    assert "质量门控审查员" in calls[1]["messages"][0]["content"]
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert "issues" not in calls[1]["messages"][0]["content"]
    assert "只输出 passed 和 reason" in calls[1]["messages"][0]["content"]
    user_content = calls[1]["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "原始报告" in user_content[0]["text"]
    assert "右肾病灶" in user_content[0]["text"]
    assert "结构化左右清单" in user_content[0]["text"]
    assert "患者右侧位于画面左侧" in user_content[0]["text"]
    assert "引导线终点" in user_content[0]["text"]
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"] == _image_data_url(b"image", "image/png")


def test_official_visual_reviewer_sends_report_and_image_together():
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    passed=False,
                    reason="报告写右肾占位，但图片将病灶画到左侧腹部；请移回患者右肾区域。",
                )
            )

    reviewer = OpenAIVisualReviewer(
        api_key="test-key",
        model="official-vision-model",
        api_style="openai",
    )
    reviewer.client = SimpleNamespace(responses=FakeResponses())

    decision = asyncio.run(
        reviewer.review(
            GenerationRequest(
                "P002",
                "右肾占位，SUVmax 8.6，考虑恶性。",
                "生成图片",
                laterality_plan=build_laterality_plan(
                    "右肾占位，SUVmax 8.6，考虑恶性。"
                ),
            ),
            GeneratedImage(b"review-image", "image/png", "fake", "fake"),
        )
    )

    assert decision.passed is False
    assert "右肾占位" in decision.summary
    assert len(calls) == 1
    assert "专门左右审查" in calls[0]["instructions"]
    content = calls[0]["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert "原始报告" in content[0]["text"]
    assert "右肾占位，SUVmax 8.6，考虑恶性。" in content[0]["text"]
    assert "结构化左右清单" in content[0]["text"]
    assert "患者右侧位于画面左侧" in content[0]["text"]
    assert "引导线终点" in content[0]["text"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == _image_data_url(b"review-image", "image/png")


def test_visual_reviewer_stops_after_laterality_precheck_failure():
    calls = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                    {
                                        "passed": False,
                                        "reason": (
                                            "【右肺门淋巴结炎性增生】报告为患者右侧；"
                                            "正面图中应位于画面左侧；"
                                            "当前引导线终点位于画面右侧。"
                                        ),
                                    },
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            )

    reviewer = OpenAIVisualReviewer(
        api_key="test-key",
        model="compatible-vision-model",
        api_style="openai_compatible",
    )
    reviewer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions())
    )

    decision = asyncio.run(
        reviewer.review(
            GenerationRequest(
                "P006",
                "右肺门淋巴结炎性增生。",
                "生成图片",
                laterality_plan=build_laterality_plan("右肺门淋巴结炎性增生。"),
            ),
            GeneratedImage(b"image", "image/png", "fake", "fake"),
        )
    )

    assert decision.passed is False
    assert "右肺门" in decision.summary
    assert len(calls) == 1


def test_visual_reviewer_ignores_generic_laterality_precheck_failure():
    calls = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                content = {
                    "passed": False,
                    "reason": (
                        "报告为患者右侧；正面图中应位于画面左侧；"
                        "当前病灶点/引导线终点实际位于画面右侧。"
                    ),
                }
            else:
                content = {"passed": True, "reason": "通过：图片与报告关键发现一致"}
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(content, ensure_ascii=False)
                        )
                    )
                ]
            )

    reviewer = OpenAIVisualReviewer(
        api_key="test-key",
        model="compatible-vision-model",
        api_style="openai_compatible",
    )
    reviewer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions())
    )

    decision = asyncio.run(
        reviewer.review(
            GenerationRequest(
                "P007",
                "右肺门淋巴结炎性增生。",
                "生成图片",
                laterality_plan=build_laterality_plan("右肺门淋巴结炎性增生。"),
            ),
            GeneratedImage(b"image", "image/png", "fake", "fake"),
        )
    )

    assert decision.passed is True
    assert len(calls) == 2


def test_gpt_image_2_edit_uses_previous_image_and_omits_input_fidelity():
    calls = []

    class FakeImages:
        def edit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(b64_json="aW1n", url=None)]
            )

    generator = OpenAIImageGenerator(
        api_key="test-key",
        model="gpt-image-2",
        base_url="https://gateway.example/v1",
        options={"input_fidelity": "high", "size": "1536x1024", "quality": "high"},
    )
    generator.client = SimpleNamespace(images=FakeImages())

    result = asyncio.run(
        generator.generate(
            GenerationRequest("P003", "右肾占位，考虑恶性。", "生成图片"),
            previous_image=GeneratedImage(b"first-image", "image/png", "fake", "fake"),
            correction_prompt="不通过理由：右肾病灶画在左侧。请用上一张图修正。",
        )
    )

    assert result.content == b"img"
    assert calls[0]["model"] == "gpt-image-2"
    assert calls[0]["image"].getvalue() == b"first-image"
    assert "input_fidelity" not in calls[0]
    assert "右肾病灶画在左侧" in calls[0]["prompt"]


def test_revision_edits_only_previous_output_not_style_reference():
    calls = []

    class FakeImages:
        def edit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json="aW1n", url=None)])

    generator = OpenAIImageGenerator(api_key="test-key", model="gpt-image-2")
    generator.client = SimpleNamespace(images=FakeImages())

    asyncio.run(
        generator.generate(
            GenerationRequest(
                "P004",
                "膀胱左后壁病灶。",
                "生成图片",
                reference_image=b"style-template",
            ),
            previous_image=GeneratedImage(
                b"first-generated-image", "image/png", "fake", "fake"
            ),
            correction_prompt="把病灶修正到患者左侧。",
        )
    )

    assert calls[0]["image"].getvalue() == b"first-generated-image"


def test_visual_reviewer_uses_explicit_frontal_laterality_mapping():
    instructions = OpenAIVisualReviewer.REVIEW_INSTRUCTIONS

    assert "患者左侧对应画面右侧" in instructions
    assert "患者右侧对应画面左侧" in instructions
    assert "病灶点或引导线端点" in instructions
