from petct.evaluation import ErrorType
from petct.providers import OpenAIVisualReviewer
from prompts import (
    IMAGE_PROMPT_VERSION,
    PET_CT_IMG2IMG_PROMPT,
    PET_CT_VISUALIZATION_PROMPT,
)


def test_image_prompt_has_a_publishable_version_identifier():
    assert IMAGE_PROMPT_VERSION == "petct-image-2026-06-23.1"


def test_visualization_prompt_prioritizes_patient_friendly_selective_anatomy():
    assert "患者友好的医学信息图" in PET_CT_VISUALIZATION_PROMPT
    assert "只绘制报告明确涉及的器官" in PET_CT_VISUALIZATION_PROMPT
    assert "不得为了“看起来完整”而画出全部内脏" in PET_CT_VISUALIZATION_PROMPT
    assert "不是绘制完整人体解剖图谱" in PET_CT_VISUALIZATION_PROMPT


def test_visualization_prompt_separates_nature_color_from_fdg_uptake():
    assert "不得仅因 FDG 摄取增高或 SUVmax 较高就把病变画成恶性红色" in PET_CT_VISUALIZATION_PROMPT
    assert "SUVmax 时，必须原样、准确地写入对应病灶标签" in PET_CT_VISUALIZATION_PROMPT
    assert "多个 SUVmax" in PET_CT_VISUALIZATION_PROMPT
    assert "不得因版面拥挤而删除关键病灶、SUVmax、侧别或性质倾向" in PET_CT_VISUALIZATION_PROMPT


def test_visualization_prompt_checks_laterality_without_explaining_it_to_patient():
    assert "仅用于内部定位核对" in PET_CT_VISUALIZATION_PROMPT
    assert "成图中不要出现 R、L、“镜像”" in PET_CT_VISUALIZATION_PROMPT
    assert "头部、胸腹部和骨盆必须保持同一正面朝向" in PET_CT_VISUALIZATION_PROMPT
    assert "先确定人体中线" in PET_CT_VISUALIZATION_PROMPT
    assert "患者右肺、右肾、右乳腺应位于画面左侧" in PET_CT_VISUALIZATION_PROMPT
    assert "患者左肺、左肾、左乳腺应位于画面右侧" in PET_CT_VISUALIZATION_PROMPT
    assert "内部侧别核对表" in PET_CT_VISUALIZATION_PROMPT
    assert "标签放在画面哪一边与病灶侧别无关" in PET_CT_VISUALIZATION_PROMPT


def test_visualization_prompt_requires_structured_finding_inventory_before_drawing():
    assert "先在内部提取一张结构化发现清单" in PET_CT_VISUALIZATION_PROMPT
    assert "每一项至少包含：原文短句、解剖部位、侧别、性质倾向、FDG/SUVmax、应使用颜色、是否需要绘制" in PET_CT_VISUALIZATION_PROMPT
    assert "清单中的每一项都必须在成图前后逐项核对" in PET_CT_VISUALIZATION_PROMPT


def test_visualization_prompt_hardens_lymph_node_anatomy_and_color_mapping():
    assert "淋巴结发现必须画成淋巴结/结节点" in PET_CT_VISUALIZATION_PROMPT
    assert "腹膜后" in PET_CT_VISUALIZATION_PROMPT
    assert "髂血管旁" in PET_CT_VISUALIZATION_PROMPT
    assert "肠系膜" in PET_CT_VISUALIZATION_PROMPT
    assert "疑似转移" in PET_CT_VISUALIZATION_PROMPT
    assert "每个病灶圆点、引导线端点和标签文字必须使用同一项发现的颜色" in PET_CT_VISUALIZATION_PROMPT


def test_reference_prompt_uses_style_without_copying_medical_content():
    assert "参考图只用于学习整体版式" in PET_CT_IMG2IMG_PROMPT
    assert "不得复制参考图中的病灶" in PET_CT_IMG2IMG_PROMPT
    assert "{report_content}" in PET_CT_IMG2IMG_PROMPT


def test_evaluation_taxonomy_covers_omission_and_anatomy():
    assert ErrorType.OMISSION.value == "OMISSION"
    assert ErrorType.ANATOMICAL_DISTORTION.value == "ANATOMICAL_DISTORTION"


def test_visual_reviewer_prompt_fails_unclear_or_non_vision_review():
    instructions = OpenAIVisualReviewer.REVIEW_INSTRUCTIONS

    assert "如果无法清楚读取图片内容或模型无法实际查看图片" in instructions
    assert "不得臆测通过" in instructions
    assert "逐项重建报告中的关键发现清单" in instructions


def test_visual_reviewer_prompt_requires_report_image_cross_check_and_veto_rules():
    instructions = OpenAIVisualReviewer.REVIEW_INSTRUCTIONS

    assert "审查输入包含两类材料：原始报告文本和待审查图片" in instructions
    assert "必须同时使用两者交叉核对" in instructions
    assert "不得只看图片的视觉效果" in instructions
    assert "只要报告中的关键发现均被正确表达" in instructions
    assert "一票否决" in instructions
    assert "只输出 passed 和 reason" in instructions
    assert "不要输出错误分类" in instructions
    assert "reason 必须写明“报告依据 + 图片问题 + 修图要求”" in instructions


def test_visual_reviewer_verifies_laterality_from_midline_and_actual_endpoint():
    instructions = OpenAIVisualReviewer.REVIEW_INSTRUCTIONS

    assert "第一步先确定人体中线和统一正面朝向" in instructions
    assert "标签位于画面哪边不能作为侧别证据" in instructions
    assert "逐项建立内部侧别核对表" in instructions
    assert "报告侧别、预期画面侧、实际落点画面侧" in instructions
    assert "患者左侧病灶实际落在画面右侧时，不得误判为左右错误" in instructions
    assert "方向不清" in instructions


def test_visual_reviewer_accepts_patient_friendly_location_granularity():
    instructions = OpenAIVisualReviewer.REVIEW_INSTRUCTIONS

    assert "患者友好粒度" in instructions
    assert "不要因为未拆成多个细分落点而判不通过" in instructions
    assert "错误侧别、错误器官大区" in instructions
    assert "门诊场景" in instructions
    assert "重大错误" in instructions
    assert "前列腺前部及两侧外周带" in instructions
    assert "两肺炎症" in instructions
