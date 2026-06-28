"""Structured patient-comprehension question generation."""

from __future__ import annotations

import asyncio
import json
import re

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator


QUESTION_PROMPT_VERSION = "petct-questions-2026-06-21.1"


_QUESTION_NUMBER_RE = re.compile(r"^\s*\d+\s*[.、:：)）]\s*")
_OPTION_PREFIX_RE = re.compile(r"^\s*[（(]?[A-Da-d][)）.、:：]\s*")
_ANSWER_LETTER_RE = re.compile(
    r"^\s*(?:正确答案\s*[为是:：]\s*)?[（(]?([A-Da-d])(?:[)）.、:：]|\s*$)"
)


def _strip_repeated_prefix(text: str, pattern: re.Pattern[str]) -> str:
    cleaned = text.strip()
    while True:
        without_prefix = pattern.sub("", cleaned, count=1).strip()
        if without_prefix == cleaned:
            return cleaned
        cleaned = without_prefix


class ComprehensionQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    answer: str
    type: str

    @model_validator(mode="after")
    def normalize_patient_question(self):
        self.question = _strip_repeated_prefix(self.question, _QUESTION_NUMBER_RE)
        original_answer = self.answer.strip()
        letter_match = _ANSWER_LETTER_RE.match(original_answer)
        cleaned_options = [
            _strip_repeated_prefix(option, _OPTION_PREFIX_RE)
            for option in self.options
        ]
        if any(not option for option in cleaned_options):
            raise ValueError("question options cannot be empty")
        if len(set(cleaned_options)) != 4:
            raise ValueError("question options must be unique")

        cleaned_answer = _strip_repeated_prefix(original_answer, _OPTION_PREFIX_RE)
        if cleaned_answer in cleaned_options:
            normalized_answer = cleaned_answer
        elif letter_match:
            normalized_answer = cleaned_options[ord(letter_match.group(1).upper()) - ord("A")]
        else:
            raise ValueError("question answer must match one option")

        self.options = cleaned_options
        self.answer = normalized_answer
        return self


class QuestionSet(BaseModel):
    questions: list[ComprehensionQuestion] = Field(min_length=2, max_length=3)


class OpenAIQuestionService:
    INSTRUCTIONS = """你是患者教育测验设计者。任务是测试患者是否记住报告原文中的关键内容；患者会先阅读 PET-CT 患者友好报告再答题。这不是医学知识考试，也不是考查医生的诊断能力。

只使用报告明确写出的事实，生成 2 至 3 道简体中文单选题，每题 4 个互斥选项且只有 1 个正确答案。

出题规则：
1. 每题都应能从报告中的一句明确原文直接找到答案，例如病灶在哪一侧/哪个部位、报告认为是什么可能、报告建议做什么检查或复查、报告写出的 SUVmax 是多少。
2. 不得要求患者推断诊断、病因、分期、预后、治疗方案或“最可能是什么”；不得用报告外医学常识才能排除干扰项。
3. 使用患者容易理解的日常表达；必要缩写应同时给出中文，例如“右上肺磨玻璃影（GGO）”。避免生僻术语和医生考试式措辞。
4. 题干不要带“1.”等题号。选项内容本身不要带 A、B、C、D 或序号，界面会统一添加选项字母。
5. 干扰项应简短、清楚，并与题干属于同一类信息；不得加入可能误导患者的新医学结论。
6. answer 必须与清理后的某一个 options 字符串完全一致。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 300,
        api_style: str = "openai",
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )
        self.model = model
        self.api_style = api_style

    async def generate(self, report_text: str) -> list[dict]:
        parsed = await asyncio.to_thread(self._generate_sync, report_text)
        if parsed is None:
            raise RuntimeError("Question model returned no structured output")
        return [question.model_dump() for question in parsed.questions]

    def _generate_sync(self, report_text: str) -> QuestionSet | None:
        if self.api_style == "openai_compatible":
            schema = json.dumps(QuestionSet.model_json_schema(), ensure_ascii=False)
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
            return QuestionSet.model_validate_json(_strip_json_fence(content))

        response = self.client.responses.parse(
            model=self.model,
            instructions=self.INSTRUCTIONS,
            input=report_text,
            text_format=QuestionSet,
        )
        return response.output_parsed


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()
