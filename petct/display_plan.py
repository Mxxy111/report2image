"""Structured display-plan contracts for PET/CT patient-facing images."""

from __future__ import annotations

from enum import StrEnum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from petct.laterality import (
    LateralityFinding,
    LateralityPlan,
    PatientSide,
    canvas_side_for,
    forbidden_canvas_side_for,
)


class DisplayFindingType(StrEnum):
    MALIGNANT = "MALIGNANT"
    BENIGN = "BENIGN"
    INDETERMINATE = "INDETERMINATE"
    IMPORTANT_NEGATIVE = "IMPORTANT_NEGATIVE"
    TREATMENT_CONTEXT = "TREATMENT_CONTEXT"


class DisplayDetailField(StrEnum):
    SUVMAX = "SUVMAX"
    FDG_UPTAKE = "FDG_UPTAKE"
    LESION_SIZE = "LESION_SIZE"
    ANATOMICAL_DETAIL = "ANATOMICAL_DETAIL"


class ReviewStrength(StrEnum):
    OFF = "OFF"
    QUICK = "QUICK"
    STANDARD = "STANDARD"
    STRICT = "STRICT"


DEFAULT_DISPLAY_FINDING_TYPES = (
    DisplayFindingType.MALIGNANT,
    DisplayFindingType.INDETERMINATE,
)
DEFAULT_DISPLAY_DETAIL_FIELDS = (
    DisplayDetailField.SUVMAX,
    DisplayDetailField.FDG_UPTAKE,
    DisplayDetailField.LESION_SIZE,
    DisplayDetailField.ANATOMICAL_DETAIL,
)


class ExcludedDisplayItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=False)

    source: Literal["conclusion", "findings"]
    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DisplayPlanItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=False)

    id: str = Field(min_length=1)
    draw: bool = True
    content_types: list[DisplayFindingType] = Field(
        default_factory=list,
        alias="contentTypes",
    )
    priority: str = "secondary"
    anatomy: str = Field(min_length=1)
    patient_side: PatientSide = Field(alias="patientSide")
    nature: str = Field(min_length=1)
    color_class: str = Field(alias="colorClass", min_length=1)
    label_text: str = Field(alias="labelText", min_length=1)
    suvmax: str | None = None
    size: str | None = None
    fdg_uptake: str | None = Field(default=None, alias="fdgUptake")
    conclusion_evidence: str | None = Field(default=None, alias="conclusionEvidence")
    findings_evidence: str | None = Field(default=None, alias="findingsEvidence")
    confidence: str = "medium"

    def evidence_texts(self) -> tuple[str, ...]:
        return tuple(
            text.strip()
            for text in (self.conclusion_evidence, self.findings_evidence)
            if text and text.strip()
        )


class DisplayPlan(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=False)

    selected_finding_types: list[DisplayFindingType] = Field(
        default_factory=list,
        alias="selectedFindingTypes",
    )
    selected_detail_fields: list[DisplayDetailField] = Field(
        default_factory=list,
        alias="selectedDetailFields",
    )
    items: list[DisplayPlanItem] = Field(default_factory=list)
    excluded_items: list[ExcludedDisplayItem] = Field(
        default_factory=list,
        alias="excludedItems",
    )
    warnings: list[str] = Field(default_factory=list)

    def drawable_items(self) -> tuple[DisplayPlanItem, ...]:
        return tuple(item for item in self.items if item.draw)

    def to_manifest(self) -> dict:
        return self.model_dump(mode="json", by_alias=True)

    def to_prompt_block(self) -> str:
        if not self.drawable_items():
            return "【展示计划】\n无可绘制项目。"
        lines = ["【展示计划】", "只绘制以下 display_plan 项目，不得新增未列出的医学发现："]
        for index, item in enumerate(self.drawable_items(), start=1):
            details = []
            if item.suvmax:
                details.append(f"SUVmax {item.suvmax}")
            if item.size:
                details.append(f"尺寸 {item.size}")
            if item.fdg_uptake:
                details.append(item.fdg_uptake)
            detail_text = f"；{'；'.join(details)}" if details else ""
            lines.append(
                f"{index}. {item.label_text}（部位：{item.anatomy}；"
                f"侧别：{item.patient_side.value}；性质：{item.nature}{detail_text}）"
            )
        return "\n".join(lines)


def normalize_display_selection(
    finding_types: list[DisplayFindingType | str] | None,
    detail_fields: list[DisplayDetailField | str] | None,
) -> tuple[tuple[DisplayFindingType, ...], tuple[DisplayDetailField, ...]]:
    normalized_findings = (
        DEFAULT_DISPLAY_FINDING_TYPES
        if finding_types is None
        else tuple(DisplayFindingType(value) for value in finding_types)
    )
    normalized_details = (
        DEFAULT_DISPLAY_DETAIL_FIELDS
        if detail_fields is None
        else tuple(DisplayDetailField(value) for value in detail_fields)
    )
    if not normalized_findings:
        raise ValueError("at least one display finding type must be selected")
    return _dedupe(normalized_findings), _dedupe(normalized_details)


def validate_display_plan_sources(
    plan: DisplayPlan,
    *,
    conclusion_text: str,
    findings_text: str | None = None,
) -> None:
    source_text = "\n".join(
        text for text in (conclusion_text, findings_text or "") if text
    )
    if not plan.drawable_items():
        raise ValueError("NO_DISPLAY_ITEMS: display plan contains no drawable items")
    for item in plan.drawable_items():
        if not item.evidence_texts():
            raise ValueError(f"display plan item {item.id} lacks evidence")
        if item.suvmax and not _value_present(item.suvmax, source_text):
            raise ValueError(f"SUVmax value for display plan item {item.id} is not present in source text")
        if item.size and not _value_present(item.size, source_text):
            raise ValueError(f"size value for display plan item {item.id} is not present in source text")


def build_laterality_plan_from_display_plan(plan: DisplayPlan) -> LateralityPlan:
    findings = []
    for item in plan.drawable_items():
        if item.patient_side == PatientSide.UNSPECIFIED:
            continue
        requires_endpoint = not (
            item.content_types
            and all(
                content_type == DisplayFindingType.TREATMENT_CONTEXT
                for content_type in item.content_types
            )
        )
        findings.append(
            LateralityFinding(
                finding=item.label_text,
                patient_side=item.patient_side,
                canvas_side=canvas_side_for(item.patient_side),
                anatomical_anchor=item.anatomy,
                forbidden_canvas_side=forbidden_canvas_side_for(item.patient_side),
                requires_endpoint=requires_endpoint,
            )
        )
    return LateralityPlan(findings=tuple(findings), source="display_plan")


def _dedupe(values):
    return tuple(dict.fromkeys(values))


def _value_present(value: str, source_text: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped in source_text:
        return True
    compact_value = re.sub(r"\s+", "", stripped)
    compact_source = re.sub(r"\s+", "", source_text)
    return compact_value in compact_source
