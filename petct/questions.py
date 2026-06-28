"""Question parsing helpers for clinical utility workflows."""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _coerce_question_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("questions"), list):
            raw_items = data["questions"]
        else:
            raw_items = next((value for value in data.values() if isinstance(value, list)), [])
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = []

    questions = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        q_type = str(item.get("type", "")).strip()
        if question and answer:
            normalized = {"question": question, "answer": answer, "type": q_type}
            options = item.get("options")
            if isinstance(options, list):
                clean_options = [str(option).strip() for option in options if str(option).strip()]
                if clean_options:
                    normalized["options"] = clean_options
            questions.append(normalized)
    return questions


def parse_generated_questions(raw: object) -> list[dict[str, Any]]:
    """Parse model-generated question JSON into a normalized list."""
    if raw is None:
        return []

    text = _strip_markdown_fence(str(raw))
    if not text:
        return []

    try:
        return _coerce_question_items(json.loads(text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            return _coerce_question_items(json.loads(match.group(0)))
        except json.JSONDecodeError:
            return []
