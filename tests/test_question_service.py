import asyncio
import json
from types import SimpleNamespace

from petct.question_service import ComprehensionQuestion, OpenAIQuestionService


def test_question_contract_removes_model_choice_labels_and_maps_letter_answer():
    question = ComprehensionQuestion(
        question="1. 报告写的病灶位于哪里？",
        options=["A. 左肾", "B、右肾", "（C）左肺", "D) 右肺"],
        answer="B. 右肾",
        type="报告记忆",
    )

    assert question.question == "报告写的病灶位于哪里？"
    assert question.options == ["左肾", "右肾", "左肺", "右肺"]
    assert question.answer == "右肾"


def test_question_prompt_targets_patient_report_memory_without_medical_reasoning():
    instructions = OpenAIQuestionService.INSTRUCTIONS

    assert "患者是否记住报告原文中的关键内容" in instructions
    assert "不是医学知识考试" in instructions
    assert "不得要求患者推断诊断" in instructions
    assert "选项内容本身不要带 A、B、C、D" in instructions


def test_compatible_question_provider_uses_chat_completions():
    calls = []
    question_set = {
        "questions": [
            {
                "question": "病灶位于哪里？",
                "options": ["左肾", "右肾", "左肺", "右肺"],
                "answer": "右肾",
                "type": "定位",
            },
            {
                "question": "报告中的 SUVmax 是多少？",
                "options": ["2.1", "4.5", "8.6", "12.0"],
                "answer": "8.6",
                "type": "定量",
            },
        ]
    }

    class FakeChatCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(question_set, ensure_ascii=False)
                        )
                    )
                ]
            )

    service = OpenAIQuestionService(
        api_key="test-key",
        model="compatible-model",
        base_url="https://gateway.example/v1",
        api_style="openai_compatible",
    )
    service.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions()),
        responses=SimpleNamespace(
            parse=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("Responses API must not be used")
            )
        ),
    )

    questions = asyncio.run(service.generate("右肾病灶，SUVmax 8.6。"))

    assert len(questions) == 2
    assert questions[0]["answer"] == "右肾"
    assert calls[0]["response_format"] == {"type": "json_object"}
