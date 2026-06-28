import json

import pytest

from petct.display_plan import (
    DEFAULT_DISPLAY_DETAIL_FIELDS,
    DEFAULT_DISPLAY_FINDING_TYPES,
    DisplayDetailField,
    DisplayFindingType,
    DisplayPlan,
    DisplayPlanItem,
    normalize_display_selection,
    validate_display_plan_sources,
)
from petct.laterality import PatientSide


CONCLUSION = "右肾占位，考虑恶性；左髂血管旁淋巴结考虑转移。"
FINDINGS = "右肾见软组织肿块，约3.2 cm，FDG代谢增高，SUVmax=12.9。"


def test_default_display_selection_is_patient_friendly_and_suvmax_ready():
    finding_types, detail_fields = normalize_display_selection(None, None)

    assert finding_types == DEFAULT_DISPLAY_FINDING_TYPES
    assert detail_fields == DEFAULT_DISPLAY_DETAIL_FIELDS
    assert finding_types == (
        DisplayFindingType.MALIGNANT,
        DisplayFindingType.INDETERMINATE,
    )
    assert DisplayDetailField.SUVMAX in detail_fields
    assert DisplayDetailField.LESION_SIZE in detail_fields


def test_display_selection_rejects_detail_only_requests():
    with pytest.raises(ValueError, match="at least one display finding type"):
        normalize_display_selection([], [DisplayDetailField.SUVMAX])


def test_display_plan_accepts_suvmax_and_size_from_source_text():
    plan = DisplayPlan(
        selectedFindingTypes=[DisplayFindingType.MALIGNANT],
        selectedDetailFields=[DisplayDetailField.SUVMAX, DisplayDetailField.LESION_SIZE],
        items=[
            DisplayPlanItem(
                id="item-1",
                contentTypes=[DisplayFindingType.MALIGNANT],
                priority="primary",
                anatomy="右肾",
                patientSide=PatientSide.RIGHT,
                nature="考虑恶性",
                colorClass="malignant_suspected",
                labelText="右肾占位，考虑恶性，SUVmax 12.9，约3.2 cm",
                suvmax="12.9",
                size="3.2 cm",
                fdgUptake="FDG代谢增高",
                conclusionEvidence="右肾占位，考虑恶性",
                findingsEvidence="右肾见软组织肿块，约3.2 cm，FDG代谢增高，SUVmax=12.9",
                confidence="high",
            )
        ],
        excludedItems=[],
        warnings=[],
    )

    validate_display_plan_sources(
        plan,
        conclusion_text=CONCLUSION,
        findings_text=FINDINGS,
    )

    encoded = json.dumps(plan.to_manifest(), ensure_ascii=False)
    assert "SUVmax 12.9" in encoded
    assert "右肾" in encoded


def test_display_plan_rejects_suvmax_not_found_verbatim_in_sources():
    plan = DisplayPlan(
        selectedFindingTypes=[DisplayFindingType.MALIGNANT],
        selectedDetailFields=[DisplayDetailField.SUVMAX],
        items=[
            DisplayPlanItem(
                id="item-1",
                contentTypes=[DisplayFindingType.MALIGNANT],
                anatomy="右肾",
                patientSide=PatientSide.RIGHT,
                nature="考虑恶性",
                colorClass="malignant_suspected",
                labelText="右肾占位，SUVmax 13.0",
                suvmax="13.0",
                conclusionEvidence="右肾占位，考虑恶性",
                findingsEvidence="右肾见软组织肿块，SUVmax=12.9",
            )
        ],
    )

    with pytest.raises(ValueError, match="SUVmax"):
        validate_display_plan_sources(
            plan,
            conclusion_text=CONCLUSION,
            findings_text=FINDINGS,
        )


def test_display_plan_rejects_size_not_found_verbatim_in_sources():
    plan = DisplayPlan(
        selectedFindingTypes=[DisplayFindingType.MALIGNANT],
        selectedDetailFields=[DisplayDetailField.LESION_SIZE],
        items=[
            DisplayPlanItem(
                id="item-1",
                contentTypes=[DisplayFindingType.MALIGNANT],
                anatomy="右肾",
                patientSide=PatientSide.RIGHT,
                nature="考虑恶性",
                colorClass="malignant_suspected",
                labelText="右肾占位，约4.0 cm",
                size="4.0 cm",
                conclusionEvidence="右肾占位，考虑恶性",
                findingsEvidence="右肾见软组织肿块，约3.2 cm",
            )
        ],
    )

    with pytest.raises(ValueError, match="size"):
        validate_display_plan_sources(
            plan,
            conclusion_text=CONCLUSION,
            findings_text=FINDINGS,
        )


def test_display_plan_rejects_items_without_evidence():
    plan = DisplayPlan(
        selectedFindingTypes=[DisplayFindingType.MALIGNANT],
        selectedDetailFields=[],
        items=[
            DisplayPlanItem(
                id="item-1",
                contentTypes=[DisplayFindingType.MALIGNANT],
                anatomy="右肾",
                patientSide=PatientSide.RIGHT,
                nature="考虑恶性",
                colorClass="malignant_suspected",
                labelText="右肾占位",
            )
        ],
    )

    with pytest.raises(ValueError, match="evidence"):
        validate_display_plan_sources(
            plan,
            conclusion_text=CONCLUSION,
            findings_text=FINDINGS,
        )
