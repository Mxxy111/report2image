import json

from petct.laterality import (
    CanvasSide,
    LateralityFinding,
    LateralityPlan,
    PatientSide,
    build_laterality_plan,
    canvas_side_for,
    merge_laterality_plans,
    resolve_patient_side,
)


CASE_2104814_REPORT = (
    "1.左侧输尿管、膀胱癌术后化疗后；"
    "膀胱左后壁不规则增厚伴FDG代谢异常增高，考虑复发，累及邻近组织可能。"
    "  2.右上肺GGO，炎症可能，建议正规抗炎后CT复查。"
    "  3.甲状腺右叶结节，FDG代谢异常增高，建议超声检查，必要时建议穿刺病理；"
    "右肺门淋巴结炎性增生；两侧睾丸少量鞘膜积液；左侧额叶及颞叶深部软化灶，建议MRI检查。"
)


def _finding_containing(plan, text):
    return next(item for item in plan.findings if text in item.anatomical_anchor)


def test_patient_side_maps_to_frontal_canvas_side():
    assert canvas_side_for(PatientSide.RIGHT) == CanvasSide.LEFT
    assert canvas_side_for(PatientSide.LEFT) == CanvasSide.RIGHT
    assert canvas_side_for(PatientSide.BILATERAL) == CanvasSide.BOTH
    assert canvas_side_for(PatientSide.MIDLINE) == CanvasSide.MIDLINE


def test_resolve_patient_side_keeps_midline_organs_unassigned():
    assert resolve_patient_side("前列腺代谢轻度增高") == PatientSide.MIDLINE
    assert resolve_patient_side("L1椎体放射性摄取增高") == PatientSide.MIDLINE
    assert resolve_patient_side("隆突下淋巴结FDG代谢增高") == PatientSide.MIDLINE


def test_case_2104814_laterality_plan_maps_key_findings():
    plan = build_laterality_plan(CASE_2104814_REPORT)

    left_ureter_history = _finding_containing(plan, "左侧输尿管")
    assert left_ureter_history.patient_side == PatientSide.LEFT
    assert left_ureter_history.canvas_side == CanvasSide.RIGHT
    assert left_ureter_history.requires_endpoint is False

    right_hilum = _finding_containing(plan, "右肺门")
    assert right_hilum.patient_side == PatientSide.RIGHT
    assert right_hilum.canvas_side == CanvasSide.LEFT
    assert right_hilum.forbidden_canvas_side == CanvasSide.RIGHT
    assert right_hilum.requires_endpoint is True

    right_thyroid = _finding_containing(plan, "甲状腺右叶")
    assert right_thyroid.patient_side == PatientSide.RIGHT
    assert right_thyroid.canvas_side == CanvasSide.LEFT
    assert right_thyroid.requires_endpoint is True

    right_upper_lung = _finding_containing(plan, "右上肺")
    assert right_upper_lung.patient_side == PatientSide.RIGHT
    assert right_upper_lung.canvas_side == CanvasSide.LEFT
    assert right_upper_lung.requires_endpoint is True

    left_bladder = _finding_containing(plan, "膀胱左后壁")
    assert left_bladder.patient_side == PatientSide.LEFT
    assert left_bladder.canvas_side == CanvasSide.RIGHT
    assert left_bladder.requires_endpoint is True

    bilateral_testes = _finding_containing(plan, "两侧睾丸")
    assert bilateral_testes.patient_side == PatientSide.BILATERAL
    assert bilateral_testes.canvas_side == CanvasSide.BOTH
    assert bilateral_testes.forbidden_canvas_side is None
    assert bilateral_testes.requires_endpoint is True


def test_laterality_prompt_block_requires_endpoint_based_review():
    plan = build_laterality_plan(CASE_2104814_REPORT)
    block = plan.to_prompt_block()

    assert "结构化左右清单" in block
    assert "患者右侧位于画面左侧" in block
    assert "患者左侧位于画面右侧" in block
    assert "标签文字可以放在任意空白区域" in block
    assert "引导线终点必须落在正确的患者侧和器官内" in block
    assert "requires_endpoint" in block
    assert "false 表示既往史或上下文" in block
    assert "右肺门" in block
    assert "canvas_side" in block


def test_laterality_plan_manifest_is_json_serializable():
    plan = build_laterality_plan(CASE_2104814_REPORT)
    encoded = json.dumps(plan.to_manifest(), ensure_ascii=False)

    assert "右肺门" in encoded
    assert "forbidden_canvas_side" in encoded


def test_merge_laterality_plans_keeps_ai_items_and_adds_script_fallbacks():
    ai_plan = LateralityPlan(
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
    script_plan = build_laterality_plan(CASE_2104814_REPORT)

    merged = merge_laterality_plans(ai_plan, script_plan)

    assert merged.source == "ai_plus_script"
    assert _finding_containing(merged, "右肺门").canvas_side == CanvasSide.LEFT
    assert _finding_containing(merged, "膀胱左后壁").canvas_side == CanvasSide.RIGHT
    assert _finding_containing(merged, "甲状腺右叶").canvas_side == CanvasSide.LEFT


def test_merge_laterality_plans_deduplicates_contained_anchors_and_promotes_endpoint():
    ai_plan = LateralityPlan(
        findings=(
            LateralityFinding(
                finding="右上肺GGO，炎症可能",
                patient_side=PatientSide.RIGHT,
                canvas_side=CanvasSide.LEFT,
                anatomical_anchor="右上肺",
                forbidden_canvas_side=CanvasSide.RIGHT,
                requires_endpoint=False,
            ),
        ),
        source="ai",
    )
    script_plan = LateralityPlan(
        findings=(
            LateralityFinding(
                finding="右上肺GGO，炎症可能",
                patient_side=PatientSide.RIGHT,
                canvas_side=CanvasSide.LEFT,
                anatomical_anchor="右上肺GGO",
                forbidden_canvas_side=CanvasSide.RIGHT,
                requires_endpoint=True,
            ),
        ),
        source="script",
    )

    merged = merge_laterality_plans(ai_plan, script_plan)

    assert len(merged.findings) == 1
    assert merged.findings[0].anatomical_anchor == "右上肺"
    assert merged.findings[0].requires_endpoint is True
