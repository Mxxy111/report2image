"""Human image-quality evaluation contract for retrospective analysis."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class OverallDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class ErrorType(StrEnum):
    LATERALITY = "LATERALITY"
    CHINESE_TEXT = "CHINESE_TEXT"
    LESION_LOCATION = "LESION_LOCATION"
    OMISSION = "OMISSION"
    ANATOMICAL_DISTORTION = "ANATOMICAL_DISTORTION"
    BENIGN_MALIGNANT = "BENIGN_MALIGNANT"
    SUVMAX = "SUVMAX"
    STYLE = "STYLE"
    HALLUCINATION = "HALLUCINATION"
    OTHER = "OTHER"


class HumanEvaluation(BaseModel):
    overallDecision: OverallDecision
    errorTypes: list[ErrorType] = Field(default_factory=list)
    reviewer: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_decision_and_errors(self):
        unique_errors = list(dict.fromkeys(self.errorTypes))
        self.errorTypes = unique_errors
        if self.overallDecision == OverallDecision.FAIL and not self.errorTypes:
            raise ValueError("failed evaluation requires at least one error type")
        if self.overallDecision == OverallDecision.PASS and self.errorTypes:
            raise ValueError("passed evaluation cannot contain error types")
        return self
