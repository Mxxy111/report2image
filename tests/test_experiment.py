import csv
import json

import pytest

from petct.experiment import (
    append_audit_event,
    build_work_items,
    export_experiment_results,
    load_completed_item_ids,
    load_evaluation_cases,
    prepare_experiment,
)


def _write_dataset(root, cancer_type, case_id):
    cancer_dir = root / cancer_type
    cancer_dir.mkdir(parents=True)
    path = cancer_dir / f"{cancer_type}_eval_sample.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "门诊号/住院号",
                "姓名",
                "PETCT检查日期",
                "检查所见",
                "检查结论",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "门诊号/住院号": case_id,
                "姓名": "不进入实验清单",
                "PETCT检查日期": "2026-01-01",
                "检查所见": "右肾见占位性病变。",
                "检查结论": "考虑恶性，建议进一步诊治。",
            }
        )
    return path


def test_load_evaluation_cases_requires_balanced_cancer_cohorts(tmp_path):
    for index, cancer_type in enumerate(("肾癌", "前列腺癌", "尿路上皮癌"), start=1):
        _write_dataset(tmp_path, cancer_type, f"P{index:03d}")

    cases, sources = load_evaluation_cases(tmp_path, expected_per_cancer=1)

    assert len(cases) == 3
    assert {case.cancer_type for case in cases} == {"肾癌", "前列腺癌", "尿路上皮癌"}
    assert all("不进入实验清单" not in case.report_text for case in cases)
    assert all(len(source.sha256) == 64 for source in sources)


def test_build_work_items_creates_deterministic_paired_strategies(tmp_path):
    for index, cancer_type in enumerate(("肾癌", "前列腺癌", "尿路上皮癌"), start=1):
        _write_dataset(tmp_path, cancer_type, f"P{index:03d}")
    cases, _ = load_evaluation_cases(tmp_path, expected_per_cancer=1)

    first = build_work_items(cases, random_seed=20260621)
    second = build_work_items(cases, random_seed=20260621)

    assert first == second
    assert len(first) == 6
    for pairing_key in {item.pairing_key for item in first}:
        assert {
            item.strategy for item in first if item.pairing_key == pairing_key
        } == {"UNGATED", "GATED"}
    assert [item.sequence for item in first] == list(range(1, 7))


def test_prepare_experiment_is_immutable_and_resumable(tmp_path):
    dataset_root = tmp_path / "dataset"
    for index, cancer_type in enumerate(("肾癌", "前列腺癌", "尿路上皮癌"), start=1):
        _write_dataset(dataset_root, cancer_type, f"P{index:03d}")
    cases, sources = load_evaluation_cases(dataset_root, expected_per_cancer=1)
    work_items = build_work_items(cases, random_seed=42)
    experiment_dir = tmp_path / "experiment"
    settings = {
        "pipelineId": "split_default",
        "maxRevisions": 1,
        "referenceImageSha256": None,
        "reviewStrength": "STANDARD",
        "displayFindingTypes": ["MALIGNANT", "INDETERMINATE"],
        "displayDetailFields": ["SUVMAX", "FDG_UPTAKE", "LESION_SIZE", "ANATOMICAL_DETAIL"],
        "inputColumns": {"findings": "检查所见", "conclusion": "检查结论"},
    }

    first = prepare_experiment(
        experiment_dir=experiment_dir,
        experiment_id="paired-v1",
        random_seed=42,
        cases=cases,
        sources=sources,
        work_items=work_items,
        settings=settings,
        project_root=tmp_path,
    )
    second = prepare_experiment(
        experiment_dir=experiment_dir,
        experiment_id="paired-v1",
        random_seed=42,
        cases=cases,
        sources=sources,
        work_items=work_items,
        settings=settings,
        project_root=tmp_path,
    )

    assert first == second
    assert first["plannedCases"] == 3
    assert first["plannedRuns"] == 6
    assert first["settings"]["reviewStrength"] == "STANDARD"
    assert "displayPlan" in first["prompts"]
    assert (experiment_dir / "work_items.csv").exists()

    with pytest.raises(ValueError, match="locked experiment"):
        prepare_experiment(
            experiment_dir=experiment_dir,
            experiment_id="paired-v1",
            random_seed=43,
            cases=cases,
            sources=sources,
            work_items=build_work_items(cases, random_seed=43),
            settings=settings,
            project_root=tmp_path,
        )


def test_completed_work_items_are_recovered_from_append_only_audit_log(tmp_path):
    audit_path = tmp_path / "events.jsonl"
    append_audit_event(audit_path, {"status": "started", "itemId": "item-1"})
    append_audit_event(
        audit_path,
        {"status": "completed", "itemId": "item-1", "runId": "a" * 32},
    )
    append_audit_event(audit_path, {"status": "failed", "itemId": "item-2"})

    assert load_completed_item_ids(audit_path) == {"item-1"}
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["timestamp"] for line in lines)


def test_export_experiment_results_includes_missing_and_completed_items(tmp_path):
    experiment_dir = tmp_path / "runtime" / "experiments" / "paired-v1"
    experiment_dir.mkdir(parents=True)
    work_items_path = experiment_dir / "work_items.csv"
    with work_items_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "item_id",
                "pairing_key",
                "case_id",
                "cancer_type",
                "strategy",
                "report_sha256",
                "source_file",
                "source_row",
            ],
        )
        writer.writeheader()
        for sequence, strategy in enumerate(("UNGATED", "GATED"), start=1):
            writer.writerow(
                {
                    "sequence": sequence,
                    "item_id": f"item-{sequence}",
                    "pairing_key": "P001:hash",
                    "case_id": "P001",
                    "cancer_type": "肾癌",
                    "strategy": strategy,
                    "report_sha256": "b" * 64,
                    "source_file": "肾癌/sample.csv",
                    "source_row": 2,
                }
            )
    (experiment_dir / "experiment_lock.json").write_text(
        json.dumps({"experimentId": "paired-v1", "plannedCases": 1, "plannedRuns": 2}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "runtime" / "cases" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "a" * 32,
                "generated_at": "2026-06-21T00:00:00+00:00",
                "duration_seconds": 12.5,
                "pairing_key": "P001:hash",
                "strategy": "UNGATED",
                "review_strength": "OFF",
                "display_selection": {
                    "finding_types": ["MALIGNANT"],
                    "detail_fields": ["SUVMAX"],
                },
                "display_plan": {
                    "items": [
                        {
                            "id": "item-1",
                            "labelText": "右肾占位，SUVmax 12.9",
                            "suvmax": "12.9",
                        }
                    ]
                },
                "accepted": None,
                "gate_outcome": {"status": "NOT_RUN", "attemptCount": 1, "revisionCount": 0},
                "pipeline": {
                    "image": {"provider": "image", "model": "image-model", "options": {}},
                    "review": {"provider": "review", "model": "review-model"},
                    "max_revisions": 1,
                },
                "reproducibility": {
                    "randomSeed": 42,
                    "providerSeedApplied": False,
                    "prompts": {
                        "image": {"version": "image-v1", "sha256": "c" * 64},
                        "displayPlan": {"version": "display-v1", "sha256": "f" * 64},
                        "review": {"version": "review-v1", "sha256": "d" * 64},
                    },
                    "software": {"sourceTreeSha256": "e" * 64},
                },
                "human_evaluation": {
                    "overallDecision": "PASS",
                    "errorTypes": [],
                    "reviewer": "doctor-a",
                    "notes": "",
                },
            }
        ),
        encoding="utf-8",
    )
    append_audit_event(
        experiment_dir / "events.jsonl",
        {
            "status": "completed",
            "itemId": "item-1",
            "runId": "a" * 32,
            "manifestPath": manifest_path.relative_to(tmp_path).as_posix(),
        },
    )
    output_path = tmp_path / "paired_runs.csv"

    rows = export_experiment_results(
        experiment_dir=experiment_dir,
        project_root=tmp_path,
        output_path=output_path,
    )

    assert [row["run_status"] for row in rows] == ["COMPLETED", "MISSING"]
    assert rows[0]["human_decision"] == "PASS"
    assert rows[0]["review_strength"] == "OFF"
    assert rows[0]["display_finding_types"] == "MALIGNANT"
    assert rows[0]["display_detail_fields"] == "SUVMAX"
    assert rows[0]["display_item_count"] == 1
    assert len(rows[0]["display_plan_sha256"]) == 64
    assert rows[0]["display_plan_prompt_version"] == "display-v1"
    assert rows[1]["human_decision"] == ""
    assert output_path.exists()
    exported = output_path.read_text(encoding="utf-8-sig")
    assert "右肾见占位性病变" not in exported
