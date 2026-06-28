import csv
import json

import pytest

from scripts.analysis.run_statistical_analysis import (
    analyze_paired_data,
    load_and_validate_paired_data,
)


FIELDNAMES = [
    "experiment_id",
    "pairing_key",
    "cancer_type",
    "strategy",
    "run_status",
    "human_decision",
    "human_error_types",
    "gate_accepted",
    "duration_seconds",
    "attempt_count",
    "revision_count",
    "image_provider",
    "image_api_style",
    "image_base_url",
    "image_timeout_seconds",
    "image_model",
    "image_options_json",
    "review_provider",
    "review_api_style",
    "review_base_url",
    "review_timeout_seconds",
    "review_model",
    "max_revisions",
    "image_prompt_version",
    "image_prompt_sha256",
    "review_prompt_version",
    "review_prompt_sha256",
    "reference_image_sha256",
    "source_tree_sha256",
    "random_seed",
    "provider_seed_applied",
]


def _write_paired_input(path):
    rows = []
    decisions = {
        "肾癌": ("FAIL", "PASS"),
        "前列腺癌": ("PASS", "PASS"),
        "尿路上皮癌": ("FAIL", "FAIL"),
    }
    for index, (cancer_type, pair) in enumerate(decisions.items(), start=1):
        for strategy, decision in zip(("UNGATED", "GATED"), pair):
            rows.append(
                {
                    "experiment_id": "paired-v1",
                    "pairing_key": f"pair-{index}",
                    "cancer_type": cancer_type,
                    "strategy": strategy,
                    "run_status": "COMPLETED",
                    "human_decision": decision,
                    "human_error_types": (
                        "LATERALITY" if decision == "FAIL" else ""
                    ),
                    "gate_accepted": (
                        "False" if strategy == "GATED" and decision == "FAIL" else "True"
                    ),
                    "duration_seconds": "10.5",
                    "attempt_count": "1",
                    "revision_count": "0",
                    "image_provider": "image-api",
                    "image_api_style": "openai_compatible",
                    "image_base_url": "https://example.com/v1",
                    "image_timeout_seconds": "300",
                    "image_model": "image-model",
                    "image_options_json": '{"quality":"high"}',
                    "review_provider": "review-api",
                    "review_api_style": "openai_compatible",
                    "review_base_url": "https://example.com/v1",
                    "review_timeout_seconds": "300",
                    "review_model": "review-model",
                    "max_revisions": "1",
                    "image_prompt_version": "image-v1",
                    "image_prompt_sha256": "b" * 64,
                    "review_prompt_version": "review-v1",
                    "review_prompt_sha256": "c" * 64,
                    "reference_image_sha256": "",
                    "source_tree_sha256": "a" * 64,
                    "random_seed": "42",
                    "provider_seed_applied": "False",
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_refuses_to_fall_back_when_real_input_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="real paired analysis input"):
        load_and_validate_paired_data(
            tmp_path / "missing.csv",
            expected_cases=3,
            expected_per_cancer=1,
        )


def test_analysis_rejects_incomplete_human_evaluation(tmp_path):
    input_path = tmp_path / "paired.csv"
    _write_paired_input(input_path)
    text = input_path.read_text(encoding="utf-8-sig")
    input_path.write_text(
        text.replace("COMPLETED,PASS", "COMPLETED,", 1),
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError, match="human PASS/FAIL"):
        load_and_validate_paired_data(
            input_path,
            expected_cases=3,
            expected_per_cancer=1,
        )


def test_paired_analysis_writes_traceable_real_results(tmp_path):
    input_path = tmp_path / "paired.csv"
    output_dir = tmp_path / "results"
    _write_paired_input(input_path)
    data = load_and_validate_paired_data(
        input_path,
        expected_cases=3,
        expected_per_cancer=1,
    )

    result = analyze_paired_data(
        data,
        input_path=input_path,
        output_dir=output_dir,
        performance_threshold=0.95,
        bootstrap_seed=20260621,
        bootstrap_iterations=1000,
    )

    assert result["sample"]["cases"] == 3
    assert result["primary"]["ungatedPassRate"] == pytest.approx(1 / 3)
    assert result["pairedComparison"]["ungatedFailGatedPass"] == 1
    assert result["pairedComparison"]["ungatedPassGatedFail"] == 0
    assert result["primary"]["thresholdMet"] is False
    saved = json.loads(
        (output_dir / "analysis_results.json").read_text(encoding="utf-8")
    )
    assert saved["provenance"]["inputSha256"]
    assert (output_dir / "pass_rates.csv").exists()
    assert (output_dir / "mcnemar.csv").exists()
