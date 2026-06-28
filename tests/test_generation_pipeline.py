import asyncio

import pytest

from petct.generation import (
    GeneratedImage,
    GenerationPipeline,
    GenerationRequest,
    PipelineStageError,
    ReviewDecision,
    ReviewIssue,
)
from petct.laterality import build_laterality_plan


class FakeGenerator:
    def __init__(self):
        self.calls = []

    async def generate(self, request, *, previous_image=None, correction_prompt=None):
        self.calls.append((previous_image, correction_prompt))
        return GeneratedImage(
            content=f"image-{len(self.calls)}".encode(),
            mime_type="image/png",
            provider="fake",
            model="fake-image",
        )


class FakeReviewer:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    async def review(self, request, image):
        return next(self.decisions)


class FailingReviewer:
    async def review(self, request, image):
        raise RuntimeError("upstream review failed")


@pytest.fixture
def generation_request():
    return GenerationRequest(case_id="P001", report_text="右肾病灶", prompt="生成医学图解")


def test_pipeline_without_gate_generates_once(generation_request):
    generator = FakeGenerator()
    result = asyncio.run(
        GenerationPipeline(generator).run(generation_request, gate_enabled=False)
    )
    assert result.accepted is None
    assert len(result.attempts) == 1
    assert len(generator.calls) == 1


def test_gate_passes_first_image_without_revision(generation_request):
    generator = FakeGenerator()
    reviewer = FakeReviewer([ReviewDecision(passed=True)])
    result = asyncio.run(
        GenerationPipeline(generator, reviewer).run(generation_request, gate_enabled=True)
    )
    assert result.accepted is True
    assert len(generator.calls) == 1


def test_gate_regenerates_with_structured_error_feedback(generation_request):
    generator = FakeGenerator()
    failed = ReviewDecision(
        passed=False,
        summary="不通过：病灶画在左侧，应为右侧。",
    )
    reviewer = FakeReviewer([failed, ReviewDecision(passed=True)])
    result = asyncio.run(
        GenerationPipeline(generator, reviewer, max_revisions=1).run(
            generation_request,
            gate_enabled=True,
        )
    )
    assert result.accepted is True
    assert len(result.attempts) == 2
    assert generator.calls[1][0].content == b"image-1"
    assert "不通过：病灶画在左侧，应为右侧。" in generator.calls[1][1]
    assert "使用上一版图片作为唯一编辑基础" in generator.calls[1][1]


def test_gate_returns_rejected_after_revision_budget_exhausted(generation_request):
    generator = FakeGenerator()
    failure = ReviewDecision(
        passed=False,
        issues=(ReviewIssue(category="text", description="中文标签不可辨认"),),
    )
    reviewer = FakeReviewer([failure, failure])
    result = asyncio.run(
        GenerationPipeline(generator, reviewer, max_revisions=1).run(
            generation_request,
            gate_enabled=True,
        )
    )
    assert result.accepted is False
    assert len(result.attempts) == 2
    assert result.final_image.content == b"image-2"
    assert result.final_image is result.attempts[-1].image


def test_correction_prompt_restates_frontal_laterality_mapping():
    prompt = ReviewDecision(
        passed=False,
        summary="报告写患者左侧，但图片位置错误。",
    ).correction_prompt()

    assert "患者左侧对应画面右侧" in prompt
    assert "患者右侧对应画面左侧" in prompt
    assert "左右修改硬规则" in prompt
    assert "只移动错误的器官、病灶点和引导线端点" in prompt
    assert "不得通过改写左右文字来掩盖位置错误" in prompt


def test_revision_prompt_includes_structured_laterality_plan():
    generator = FakeGenerator()
    failed = ReviewDecision(
        passed=False,
        summary="右肺门淋巴结落点位于画面右侧。",
    )
    request = GenerationRequest(
        case_id="P005",
        report_text="右肺门淋巴结炎性增生。",
        prompt="生成医学图解",
        laterality_plan=build_laterality_plan("右肺门淋巴结炎性增生。"),
    )

    asyncio.run(
        GenerationPipeline(
            generator,
            FakeReviewer([failed, ReviewDecision(passed=True)]),
            max_revisions=1,
        ).run(request, gate_enabled=True)
    )

    correction_prompt = generator.calls[1][1]
    assert "结构化左右清单" in correction_prompt
    assert "右肺门" in correction_prompt
    assert "患者右侧位于画面左侧" in correction_prompt
    assert "引导线终点" in correction_prompt


def test_pipeline_reports_real_generation_and_review_progress(generation_request):
    events = []

    async def capture(event):
        events.append(event)

    result = asyncio.run(
        GenerationPipeline(
            FakeGenerator(),
            FakeReviewer([ReviewDecision(passed=True)]),
        ).run(
            generation_request,
            gate_enabled=True,
            progress_callback=capture,
        )
    )

    assert result.accepted is True
    assert [(event["stage"], event["status"]) for event in events] == [
        ("image", "running"),
        ("image", "completed"),
        ("review", "running"),
        ("review", "completed"),
    ]


def test_pipeline_wraps_reviewer_failure_with_stage(generation_request):
    with pytest.raises(PipelineStageError) as exc_info:
        asyncio.run(
            GenerationPipeline(FakeGenerator(), FailingReviewer()).run(
                generation_request,
                gate_enabled=True,
            )
        )

    assert exc_info.value.stage == "review"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
