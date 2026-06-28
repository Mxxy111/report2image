from pathlib import Path

from petct.data import (
    build_record_key,
    detect_id_column,
    detect_text_columns,
    extract_image_record_id,
)
from petct.questions import parse_generated_questions


def test_detect_id_column_prefers_requested_column():
    columns = ["姓名", "门诊号/住院号", "ID"]
    assert detect_id_column(columns, requested="ID") == "ID"


def test_detect_id_column_falls_back_to_medical_id_column():
    columns = ["姓名", "门诊号/住院号", "检查结论"]
    assert detect_id_column(columns) == "门诊号/住院号"


def test_detect_text_columns_uses_requested_columns_in_order():
    columns = ["检查所见", "检查结论", "诊断"]
    assert detect_text_columns(columns, requested=["检查结论", "检查所见"]) == [
        "检查结论",
        "检查所见",
    ]


def test_extract_image_record_id_preserves_underscores_in_original_id():
    assert extract_image_record_id("abc_def_123456.png") == "abc_def"


def test_extract_image_record_id_keeps_full_stem_when_suffix_is_not_hash():
    assert extract_image_record_id(Path("abc_def_report.png")) == "abc_def_report"


def test_build_record_key_changes_when_same_patient_has_different_report():
    first = build_record_key("P001", "右肾术后，未见复发。")
    second = build_record_key("P001", "右肾术后，肺部新增结节。")
    assert first != second
    assert first.startswith("P001:")


def test_parse_generated_questions_accepts_json_list():
    raw = '[{"question":"是否有骨转移？","answer":"否","type":"存在性"}]'
    assert parse_generated_questions(raw) == [
        {"question": "是否有骨转移？", "answer": "否", "type": "存在性"}
    ]


def test_parse_generated_questions_accepts_json_object_with_questions_key():
    raw = '{"questions":[{"question":"SUVmax是多少？","answer":"6.2","type":"定量"}]}'
    assert parse_generated_questions(raw) == [
        {"question": "SUVmax是多少？", "answer": "6.2", "type": "定量"}
    ]


def test_parse_generated_questions_strips_markdown_fence():
    raw = '```json\n{"questions":[{"question":"病灶在哪里？","answer":"左肺","type":"定位"}]}\n```'
    assert parse_generated_questions(raw)[0]["answer"] == "左肺"


def test_parse_generated_questions_returns_empty_list_for_invalid_json():
    assert parse_generated_questions("not json") == []


def test_parse_generated_questions_preserves_multiple_choice_options():
    raw = (
        '{"questions":[{"question":"病灶位于哪侧？",'
        '"options":["左侧","右侧","双侧","报告未提及"],'
        '"answer":"右侧","type":"定位"}]}'
    )
    assert parse_generated_questions(raw)[0]["options"] == [
        "左侧",
        "右侧",
        "双侧",
        "报告未提及",
    ]
