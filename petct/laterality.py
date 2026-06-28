"""Deterministic laterality planning for frontal PET-CT visualizations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import replace
from enum import StrEnum


LATERALITY_PLAN_VERSION = "petct-laterality-2026-06-21.1"


class PatientSide(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BILATERAL = "BILATERAL"
    MIDLINE = "MIDLINE"
    UNSPECIFIED = "UNSPECIFIED"


class CanvasSide(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BOTH = "BOTH"
    MIDLINE = "MIDLINE"
    UNASSIGNED = "UNASSIGNED"


_CLAUSE_SPLIT_RE = re.compile(r"[。；;\n]+")
_LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[.、:：)）]\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_PREFIX_SIDE_RE = re.compile(r"(?P<side>左|右)(?:侧)?(?P<body>[^，,；;。]{1,28})")
_BILATERAL_RE = re.compile(
    r"(?P<anchor>(?:双侧|两侧|双|两)"
    r"(?:肺门|肺|肾上腺|肾|乳腺|睾丸|精囊腺|腋窝|颈部|锁骨上|上颌窦|筛窦|蝶窦|"
    r"附件|髂血管旁|腹股沟|肋骨|肩|髂骨|输尿管|卵巢)"
    r"[^，,；;。]{0,24})"
)
_INFIX_SIDE_RE = re.compile(
    r"(?P<anchor>"
    r"(?:甲状腺|膀胱|肝|肺|肾上腺|肾|乳腺|输尿管|脑|额叶|颞叶|枕叶|"
    r"髂血管旁|髂骨|股骨|肋骨|骶髂关节|上颌窦|咽隐窝|眼球)"
    r"[^，,；;。]{0,8}(?P<side>左|右)[^，,；;。]{1,24})"
)
_MIDLINE_PATTERNS = (
    re.compile(r"前列腺"),
    re.compile(r"隆突下"),
    re.compile(r"纵隔"),
    re.compile(r"中线"),
    re.compile(r"脊柱"),
    re.compile(r"椎体"),
    re.compile(r"[CLTS]\d+(?:/S\d+)?"),
    re.compile(r"膀胱(?!左|右)"),
    re.compile(r"子宫(?!左|右)"),
    re.compile(r"胰腺(?!左|右)"),
    re.compile(r"胃"),
    re.compile(r"食管"),
)
_SIGNIFICANT_TERMS = (
    "病灶",
    "占位",
    "结节",
    "增厚",
    "代谢",
    "摄取",
    "SUV",
    "转移",
    "复发",
    "炎性",
    "增生",
    "GGO",
    "软化灶",
    "积液",
    "破坏",
    "囊肿",
    "术后",
)
_VISUAL_ENDPOINT_TERMS = (
    "病灶",
    "占位",
    "结节",
    "增厚",
    "代谢",
    "摄取",
    "SUV",
    "转移",
    "复发",
    "炎性",
    "增生",
    "GGO",
    "软化灶",
    "积液",
    "破坏",
    "囊肿",
    "淋巴结",
    "肿块",
    "异常",
)


@dataclass(frozen=True)
class LateralityFinding:
    finding: str
    patient_side: PatientSide
    canvas_side: CanvasSide
    anatomical_anchor: str
    forbidden_canvas_side: CanvasSide | None = None
    requires_endpoint: bool = True

    def to_manifest(self) -> dict[str, str | bool | None]:
        return {
            "finding": self.finding,
            "patient_side": self.patient_side.value,
            "canvas_side": self.canvas_side.value,
            "anatomical_anchor": self.anatomical_anchor,
            "forbidden_canvas_side": (
                self.forbidden_canvas_side.value
                if self.forbidden_canvas_side
                else None
            ),
            "requires_endpoint": self.requires_endpoint,
        }


@dataclass(frozen=True)
class LateralityPlan:
    findings: tuple[LateralityFinding, ...]
    version: str = LATERALITY_PLAN_VERSION
    source: str = "script"

    def to_manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source": self.source,
            "rules": {
                "frontal_view": "患者右侧位于画面左侧；患者左侧位于画面右侧。",
                "endpoint_check": "标签文字位置不能作为侧别依据；必须检查病灶点或引导线终点。",
            },
            "findings": [finding.to_manifest() for finding in self.findings],
        }

    def to_prompt_block(self) -> str:
        if not self.findings:
            return (
                "【结构化左右清单】\n"
                "本报告未提取到明确左/右/双侧或中线定位项；不要自行添加侧别。"
            )
        payload = json.dumps(
            [finding.to_manifest() for finding in self.findings],
            ensure_ascii=False,
            indent=2,
        )
        return (
            "【结构化左右清单】\n"
            "下面 JSON 是程序在生成前固定下来的侧别计划，必须优先于自由排版习惯执行。\n"
            "固定规则：患者右侧位于画面左侧；患者左侧位于画面右侧；"
            "双侧发现必须在两侧对称或分别呈现；中线器官不分配左右侧。\n"
            "标签文字可以放在任意空白区域，但病灶点和引导线终点必须落在正确的患者侧和器官内；"
            "不能只看标签文字中的“左/右”来判断正确。"
            "requires_endpoint=false 表示既往史或上下文，不要求新增病灶点；"
            "requires_endpoint=true 的条目必须有可核对的器官/病灶点/引导线终点。\n"
            f"{payload}"
        )

    def to_review_block(self) -> str:
        if not self.findings:
            return "结构化左右清单：本报告未提取到明确侧别项；如图片自行添加左右侧发现，应判为不通过。"
        return (
            f"{self.to_prompt_block()}\n"
            "审查时请逐条填写内部核对：报告患者侧别、预期画面侧、实际引导线终点画面侧、"
            "实际器官/区域。任何一条无法确认或违反映射，都必须判为不通过。"
        )

    def to_correction_block(self) -> str:
        if not self.findings:
            return ""
        return (
            f"{self.to_prompt_block()}\n"
            "重修时只移动违反上述清单的病灶点、器官局部或引导线终点；"
            "保留已经正确的发现、标签、颜色和 SUVmax。"
        )


def canvas_side_for(patient_side: PatientSide) -> CanvasSide:
    return {
        PatientSide.RIGHT: CanvasSide.LEFT,
        PatientSide.LEFT: CanvasSide.RIGHT,
        PatientSide.BILATERAL: CanvasSide.BOTH,
        PatientSide.MIDLINE: CanvasSide.MIDLINE,
    }.get(patient_side, CanvasSide.UNASSIGNED)


def forbidden_canvas_side_for(patient_side: PatientSide) -> CanvasSide | None:
    if patient_side == PatientSide.RIGHT:
        return CanvasSide.RIGHT
    if patient_side == PatientSide.LEFT:
        return CanvasSide.LEFT
    return None


def resolve_patient_side(text: str) -> PatientSide:
    normalized = _normalize_text(text)
    has_left = "左" in normalized
    has_right = "右" in normalized
    if _BILATERAL_RE.search(normalized) or (has_left and has_right):
        return PatientSide.BILATERAL
    if has_right:
        return PatientSide.RIGHT
    if has_left:
        return PatientSide.LEFT
    if any(pattern.search(normalized) for pattern in _MIDLINE_PATTERNS):
        return PatientSide.MIDLINE
    return PatientSide.UNSPECIFIED


def build_laterality_plan(report_text: str) -> LateralityPlan:
    findings: list[LateralityFinding] = []
    seen: set[tuple[str, PatientSide]] = set()

    for clause in _iter_clauses(report_text):
        for anchor, side in _extract_side_anchors(clause):
            _append_finding(findings, seen, clause, anchor, side)

        if not any(clause == finding.finding for finding in findings):
            side = resolve_patient_side(clause)
            if side == PatientSide.MIDLINE and _is_significant_clause(clause):
                _append_finding(findings, seen, clause, _midline_anchor(clause), side)

    return LateralityPlan(findings=tuple(findings), source="script")


def merge_laterality_plans(
    primary: LateralityPlan,
    fallback: LateralityPlan,
) -> LateralityPlan:
    merged: list[LateralityFinding] = list(primary.findings)
    for finding in fallback.findings:
        existing_index = _matching_finding_index(merged, finding)
        if existing_index is None:
            merged.append(finding)
        else:
            existing = merged[existing_index]
            if finding.requires_endpoint and not existing.requires_endpoint:
                merged[existing_index] = replace(existing, requires_endpoint=True)
    source = (
        f"{primary.source}_plus_{fallback.source}"
        if primary.source != fallback.source
        else primary.source
    )
    return LateralityPlan(
        findings=tuple(merged),
        version=primary.version,
        source=source,
    )


def _matching_finding_index(
    findings: list[LateralityFinding],
    candidate: LateralityFinding,
) -> int | None:
    candidate_anchor = _identity_text(candidate.anatomical_anchor)
    for index, finding in enumerate(findings):
        if finding.patient_side != candidate.patient_side:
            continue
        anchor = _identity_text(finding.anatomical_anchor)
        if (
            anchor == candidate_anchor
            or (anchor and candidate_anchor and anchor in candidate_anchor)
            or (anchor and candidate_anchor and candidate_anchor in anchor)
        ):
            return index
    return None


def _append_finding(
    findings: list[LateralityFinding],
    seen: set[tuple[str, PatientSide]],
    clause: str,
    anchor: str,
    side: PatientSide,
) -> None:
    clean_anchor = _clean_anchor(anchor)
    if not clean_anchor:
        return
    key = (clean_anchor, side)
    if key in seen:
        return
    seen.add(key)
    findings.append(
            LateralityFinding(
                finding=clause,
                patient_side=side,
                canvas_side=canvas_side_for(side),
                anatomical_anchor=clean_anchor,
                forbidden_canvas_side=forbidden_canvas_side_for(side),
                requires_endpoint=_requires_endpoint(clause, clean_anchor),
            )
    )


def _extract_side_anchors(clause: str) -> list[tuple[str, PatientSide]]:
    matches: list[tuple[int, int, str, PatientSide]] = []
    for match in _INFIX_SIDE_RE.finditer(clause):
        side = PatientSide.RIGHT if match.group("side") == "右" else PatientSide.LEFT
        matches.append((match.start(), match.end(), match.group("anchor"), side))
    for match in _BILATERAL_RE.finditer(clause):
        matches.append((match.start(), match.end(), match.group("anchor"), PatientSide.BILATERAL))
    for match in _PREFIX_SIDE_RE.finditer(clause):
        if _span_inside_existing(match.start(), match.end(), matches):
            continue
        side = PatientSide.RIGHT if match.group("side") == "右" else PatientSide.LEFT
        matches.append((match.start(), match.end(), match.group(0), side))

    matches.sort(key=lambda item: item[0])
    return [(anchor, side) for _, _, anchor, side in matches]


def _span_inside_existing(
    start: int,
    end: int,
    matches: list[tuple[int, int, str, PatientSide]],
) -> bool:
    return any(existing_start <= start and end <= existing_end for existing_start, existing_end, _, _ in matches)


def _iter_clauses(report_text: str) -> list[str]:
    clauses = []
    for raw_clause in _CLAUSE_SPLIT_RE.split(report_text):
        clause = _LEADING_NUMBER_RE.sub("", _normalize_text(raw_clause))
        if clause:
            clauses.append(clause)
    return clauses


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub("", text.strip())


def _identity_text(text: str) -> str:
    return _normalize_text(text).strip("，,；;。:：、")


def _clean_anchor(anchor: str) -> str:
    cleaned = _normalize_text(anchor).strip("，,；;。:：、")
    return cleaned[:40]


def _is_significant_clause(clause: str) -> bool:
    return any(term in clause for term in _SIGNIFICANT_TERMS)


def _requires_endpoint(clause: str, anchor: str) -> bool:
    haystack = f"{clause}{anchor}"
    return any(term in haystack for term in _VISUAL_ENDPOINT_TERMS)


def _midline_anchor(clause: str) -> str:
    for pattern in _MIDLINE_PATTERNS:
        match = pattern.search(clause)
        if match:
            return match.group(0)
    return clause[:40]
