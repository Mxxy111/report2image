import pytest
from pydantic import ValidationError

from petct.evaluation import ErrorType, HumanEvaluation, OverallDecision


def test_failed_evaluation_requires_at_least_one_error_type():
    with pytest.raises(ValidationError):
        HumanEvaluation(
            overallDecision=OverallDecision.FAIL,
            errorTypes=[],
        )


def test_passed_evaluation_cannot_contain_error_types():
    with pytest.raises(ValidationError):
        HumanEvaluation(
            overallDecision=OverallDecision.PASS,
            errorTypes=[ErrorType.LATERALITY],
        )


def test_failed_evaluation_supports_multiple_error_types():
    evaluation = HumanEvaluation(
        overallDecision=OverallDecision.FAIL,
        errorTypes=[ErrorType.LATERALITY, ErrorType.CHINESE_TEXT],
        reviewer="doctor-a",
    )
    assert evaluation.errorTypes == [
        ErrorType.LATERALITY,
        ErrorType.CHINESE_TEXT,
    ]
