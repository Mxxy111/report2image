"""Provider-neutral image generation and visual-gating domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
from typing import Protocol

from petct.display_plan import DisplayPlan
from petct.laterality import LateralityPlan


ProgressCallback = Callable[[dict[str, object]], Awaitable[None]]


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class GenerationRequest:
    case_id: str
    report_text: str
    prompt: str
    reference_image: bytes | None = None
    reference_mime_type: str = "image/png"
    laterality_plan: LateralityPlan | None = None
    display_plan: DisplayPlan | None = None


@dataclass(frozen=True)
class GeneratedImage:
    content: bytes
    mime_type: str
    provider: str
    model: str


@dataclass(frozen=True)
class ReviewIssue:
    category: str
    description: str
    severity: str = "major"


@dataclass(frozen=True)
class ReviewDecision:
    passed: bool
    issues: tuple[ReviewIssue, ...] = ()
    summary: str = ""

    def correction_prompt(self, laterality_plan: LateralityPlan | None = None) -> str:
        if self.passed:
            return ""
        reason = self.summary.strip()
        if not reason:
            reason = "\n".join(
                f"- {issue.description}"
                for issue in self.issues
            ).strip()
        if not reason:
            reason = "AI 门控判定不通过，但未返回具体原因；请重新核对报告与图片。"
        laterality_block = (
            f"\n{laterality_plan.to_correction_block()}\n"
            if laterality_plan
            else ""
        )
        return (
            "使用上一版图片作为唯一编辑基础，调用 image edit 进行局部修订；"
            "不要从零重画，不要改变未提及且正确的内容。\n"
            "【左右修改硬规则】保持正面面对患者的统一视角：患者左侧对应画面右侧，"
            "患者右侧对应画面左侧。先确定人体中线，再按器官、病灶点和引导线端点的实际落点执行；"
            "标签位于画面哪边不能作为侧别依据。若反馈涉及左右错误，只移动错误的器官、病灶点和引导线端点，"
            "不得通过改写左右文字来掩盖位置错误，也不得移动其他已正确发现。\n"
            f"{laterality_block}"
            f"AI 门控不通过理由：{reason}\n"
            "请只修复上述理由中指出的问题，并再次逐项核对左右侧、中文文字、病灶位置、"
            "良恶性颜色、FDG/SUVmax、引导线和患者友好风格。"
        )


@dataclass(frozen=True)
class GenerationAttempt:
    image: GeneratedImage
    review: ReviewDecision | None


@dataclass(frozen=True)
class PipelineResult:
    final_image: GeneratedImage
    accepted: bool | None
    gate_enabled: bool
    attempts: tuple[GenerationAttempt, ...] = field(default_factory=tuple)


class ImageGenerator(Protocol):
    async def generate(
        self,
        request: GenerationRequest,
        *,
        previous_image: GeneratedImage | None = None,
        correction_prompt: str | None = None,
    ) -> GeneratedImage: ...


class ImageReviewer(Protocol):
    async def review(
        self,
        request: GenerationRequest,
        image: GeneratedImage,
    ) -> ReviewDecision: ...


class GenerationPipeline:
    def __init__(
        self,
        generator: ImageGenerator,
        reviewer: ImageReviewer | None = None,
        max_revisions: int = 1,
    ):
        if max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        self.generator = generator
        self.reviewer = reviewer
        self.max_revisions = max_revisions

    async def run(
        self,
        request: GenerationRequest,
        gate_enabled: bool,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        if gate_enabled and self.reviewer is None:
            raise ValueError("gate_enabled requires an image reviewer")

        await _notify_progress(
            progress_callback,
            stage="image",
            status="running",
            message="正在调用生图模型",
            attempt=1,
        )
        try:
            image = await self.generator.generate(request)
        except Exception as exc:
            raise PipelineStageError("image", "image generation failed") from exc
        await _notify_progress(
            progress_callback,
            stage="image",
            status="completed",
            message="图片生成完成",
            attempt=1,
        )
        if not gate_enabled:
            attempt = GenerationAttempt(image=image, review=None)
            return PipelineResult(
                final_image=image,
                accepted=None,
                gate_enabled=False,
                attempts=(attempt,),
            )

        attempts = []
        for revision_index in range(self.max_revisions + 1):
            await _notify_progress(
                progress_callback,
                stage="review",
                status="running",
                message="正在进行 AI 图片审查",
                attempt=revision_index + 1,
            )
            try:
                decision = await self.reviewer.review(request, image)
            except Exception as exc:
                raise PipelineStageError("review", "image review failed") from exc
            attempts.append(GenerationAttempt(image=image, review=decision))
            await _notify_progress(
                progress_callback,
                stage="review",
                status="completed",
                message="AI 图片审查完成",
                attempt=revision_index + 1,
                passed=decision.passed,
                issueCount=len(decision.issues),
            )
            if decision.passed:
                return PipelineResult(
                    final_image=image,
                    accepted=True,
                    gate_enabled=True,
                    attempts=tuple(attempts),
                )
            if revision_index < self.max_revisions:
                await _notify_progress(
                    progress_callback,
                    stage="image",
                    status="running",
                    message="正在根据审查意见修订图片",
                    attempt=revision_index + 2,
                )
                try:
                    image = await self.generator.generate(
                        request,
                        previous_image=image,
                        correction_prompt=decision.correction_prompt(
                            request.laterality_plan
                        ),
                    )
                except Exception as exc:
                    raise PipelineStageError("image", "image revision failed") from exc
                await _notify_progress(
                    progress_callback,
                    stage="image",
                    status="completed",
                    message="图片修订完成",
                    attempt=revision_index + 2,
                )

        return PipelineResult(
            final_image=image,
            accepted=False,
            gate_enabled=True,
            attempts=tuple(attempts),
        )


async def _notify_progress(
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
