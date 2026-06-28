"""Strict real-data analysis for the frozen 300-case paired experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from petct.provenance import sha256_file  # noqa: E402


ANALYSIS_VERSION = "paired-sap-2.0"
CANCER_TYPES = ("肾癌", "前列腺癌", "尿路上皮癌")
STRATEGIES = ("UNGATED", "GATED")
REQUIRED_COLUMNS = {
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
}
FROZEN_COLUMNS = (
    "experiment_id",
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
)


def load_and_validate_paired_data(
    input_path: str | Path,
    *,
    expected_cases: int = 300,
    expected_per_cancer: int = 100,
) -> pd.DataFrame:
    """Load only a complete, human-rated, frozen real experiment export."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"real paired analysis input not found: {path}; "
            "simulation fallback is intentionally disabled"
        )
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"analysis input is missing columns: {missing_columns}")
    if len(data) != expected_cases * 2:
        raise ValueError(
            f"expected {expected_cases * 2} planned runs; found {len(data)}"
        )
    if set(data["run_status"]) != {"COMPLETED"}:
        counts = data["run_status"].value_counts().to_dict()
        raise ValueError(f"all runs must be COMPLETED before analysis: {counts}")
    if not data["human_decision"].isin(["PASS", "FAIL"]).all():
        counts = data["human_decision"].value_counts(dropna=False).to_dict()
        raise ValueError(
            f"every completed run requires a human PASS/FAIL decision: {counts}"
        )
    if data.duplicated(["pairing_key", "strategy"]).any():
        raise ValueError("duplicate pairing_key/strategy rows found")
    pair_sizes = data.groupby("pairing_key")["strategy"].agg(
        lambda values: frozenset(values)
    )
    if len(pair_sizes) != expected_cases or not all(
        strategies == frozenset(STRATEGIES) for strategies in pair_sizes
    ):
        raise ValueError("each case must have exactly one UNGATED and one GATED run")
    cancer_by_pair = data.groupby("pairing_key")["cancer_type"].nunique()
    if (cancer_by_pair != 1).any():
        raise ValueError("paired runs disagree on cancer type")
    case_counts = (
        data.drop_duplicates("pairing_key")["cancer_type"].value_counts().to_dict()
    )
    expected_counts = {cancer: expected_per_cancer for cancer in CANCER_TYPES}
    if case_counts != expected_counts:
        raise ValueError(
            f"expected balanced cancer cohorts {expected_counts}; found {case_counts}"
        )
    for column in FROZEN_COLUMNS:
        values = data[column].unique()
        if len(values) != 1:
            raise ValueError(
                f"frozen experiment column {column!r} has {len(values)} values"
            )
    return data


def _wilson(successes: int, total: int) -> tuple[float, float]:
    low, high = proportion_confint(successes, total, alpha=0.05, method="wilson")
    return float(low), float(high)


def _bootstrap_paired_difference(
    ungated: np.ndarray,
    gated: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = gated - ungated
    indexes = rng.integers(0, len(differences), size=(iterations, len(differences)))
    estimates = differences[indexes].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _pass_rate_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    scopes = [("OVERALL", data)]
    scopes.extend((cancer, data[data["cancer_type"] == cancer]) for cancer in CANCER_TYPES)
    for scope, scoped in scopes:
        for strategy in STRATEGIES:
            selected = scoped[scoped["strategy"] == strategy]
            successes = int((selected["human_decision"] == "PASS").sum())
            total = len(selected)
            low, high = _wilson(successes, total)
            rows.append(
                {
                    "scope": scope,
                    "strategy": strategy,
                    "pass_count": successes,
                    "total": total,
                    "pass_rate": successes / total,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return rows


def _paired_rows(
    data: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> list[dict[str, object]]:
    rows = []
    scopes = [("OVERALL", data)]
    scopes.extend((cancer, data[data["cancer_type"] == cancer]) for cancer in CANCER_TYPES)
    for scope_index, (scope, scoped) in enumerate(scopes):
        wide = scoped.pivot(
            index="pairing_key",
            columns="strategy",
            values="human_decision",
        )
        ungated = (wide["UNGATED"] == "PASS").astype(int).to_numpy()
        gated = (wide["GATED"] == "PASS").astype(int).to_numpy()
        both_fail = int(((ungated == 0) & (gated == 0)).sum())
        ungated_fail_gated_pass = int(((ungated == 0) & (gated == 1)).sum())
        ungated_pass_gated_fail = int(((ungated == 1) & (gated == 0)).sum())
        both_pass = int(((ungated == 1) & (gated == 1)).sum())
        test = mcnemar(
            [
                [both_fail, ungated_fail_gated_pass],
                [ungated_pass_gated_fail, both_pass],
            ],
            exact=True,
        )
        ci_low, ci_high = _bootstrap_paired_difference(
            ungated,
            gated,
            seed=bootstrap_seed + scope_index,
            iterations=bootstrap_iterations,
        )
        rows.append(
            {
                "scope": scope,
                "cases": len(wide),
                "both_fail": both_fail,
                "ungated_fail_gated_pass": ungated_fail_gated_pass,
                "ungated_pass_gated_fail": ungated_pass_gated_fail,
                "both_pass": both_pass,
                "paired_rate_difference": float(gated.mean() - ungated.mean()),
                "difference_ci95_low_bootstrap": ci_low,
                "difference_ci95_high_bootstrap": ci_high,
                "mcnemar_exact_p": float(test.pvalue),
            }
        )
    return rows


def _error_type_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for strategy in STRATEGIES:
        strategy_data = data[data["strategy"] == strategy]
        counts: Counter[str] = Counter()
        for value in strategy_data["human_error_types"]:
            counts.update(error for error in value.split("|") if error)
        for error_type, count in sorted(counts.items()):
            rows.append(
                {
                    "strategy": strategy,
                    "error_type": error_type,
                    "count": count,
                    "rate_per_run": count / len(strategy_data),
                }
            )
    return rows


def _gate_diagnostic(data: pd.DataFrame) -> dict[str, object] | None:
    gated = data[data["strategy"] == "GATED"].copy()
    normalized = gated["gate_accepted"].str.lower()
    if not normalized.isin(["true", "false"]).all():
        return None
    ai_detected_failure = normalized == "false"
    human_failure = gated["human_decision"] == "FAIL"
    true_positive = int((ai_detected_failure & human_failure).sum())
    false_positive = int((ai_detected_failure & ~human_failure).sum())
    true_negative = int((~ai_detected_failure & ~human_failure).sum())
    false_negative = int((~ai_detected_failure & human_failure).sum())

    def safe_ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "definition": "AI final gate failure is positive; human final-image FAIL is reference",
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "trueNegative": true_negative,
        "falseNegative": false_negative,
        "sensitivity": safe_ratio(true_positive, true_positive + false_negative),
        "specificity": safe_ratio(true_negative, true_negative + false_positive),
        "positivePredictiveValue": safe_ratio(
            true_positive, true_positive + false_positive
        ),
        "negativePredictiveValue": safe_ratio(
            true_negative, true_negative + false_negative
        ),
    }


def analyze_paired_data(
    data: pd.DataFrame,
    *,
    input_path: str | Path,
    output_dir: str | Path,
    performance_threshold: float = 0.95,
    bootstrap_seed: int = 20260621,
    bootstrap_iterations: int = 10000,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pass_rates = _pass_rate_rows(data)
    paired = _paired_rows(
        data,
        bootstrap_seed=bootstrap_seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    errors = _error_type_rows(data)
    overall_rates = {
        row["strategy"]: row for row in pass_rates if row["scope"] == "OVERALL"
    }
    overall_paired = next(row for row in paired if row["scope"] == "OVERALL")
    ungated_rate = float(overall_rates["UNGATED"]["pass_rate"])
    result = {
        "analysisVersion": ANALYSIS_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "inputPath": str(Path(input_path)),
            "inputSha256": sha256_file(input_path),
            "experimentId": data["experiment_id"].iloc[0],
            "sourceTreeSha256": data["source_tree_sha256"].iloc[0],
            "randomSeed": data["random_seed"].iloc[0],
            "bootstrapSeed": bootstrap_seed,
            "bootstrapIterations": bootstrap_iterations,
        },
        "sample": {
            "cases": int(data["pairing_key"].nunique()),
            "runs": len(data),
            "cancerCounts": (
                data.drop_duplicates("pairing_key")["cancer_type"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
        },
        "configuration": {
            column: data[column].iloc[0] for column in FROZEN_COLUMNS
        },
        "primary": {
            "endpoint": "Human PASS proportion for the UNGATED one-shot image",
            "ungatedPassRate": ungated_rate,
            "ci95LowWilson": overall_rates["UNGATED"]["ci95_low"],
            "ci95HighWilson": overall_rates["UNGATED"]["ci95_high"],
            "performanceThreshold": performance_threshold,
            "thresholdRule": "point estimate greater than or equal to threshold",
            "thresholdMet": ungated_rate >= performance_threshold,
        },
        "pairedComparison": {
            "gatedPassRate": overall_rates["GATED"]["pass_rate"],
            "pairedRateDifference": overall_paired["paired_rate_difference"],
            "differenceCi95LowBootstrap": overall_paired[
                "difference_ci95_low_bootstrap"
            ],
            "differenceCi95HighBootstrap": overall_paired[
                "difference_ci95_high_bootstrap"
            ],
            "ungatedFailGatedPass": overall_paired[
                "ungated_fail_gated_pass"
            ],
            "ungatedPassGatedFail": overall_paired[
                "ungated_pass_gated_fail"
            ],
            "mcnemarExactP": overall_paired["mcnemar_exact_p"],
        },
        "gateDiagnostic": _gate_diagnostic(data),
        "limitations": [
            "The image provider does not expose a seed, so randomSeed does not "
            "make image pixels deterministic.",
            "The primary threshold uses the observed point estimate; its Wilson "
            "95% confidence interval is reported as uncertainty.",
        ],
    }

    pd.DataFrame(pass_rates).to_csv(
        destination / "pass_rates.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(paired).to_csv(
        destination / "mcnemar.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(errors).to_csv(
        destination / "error_types.csv", index=False, encoding="utf-8-sig"
    )
    (destination / "analysis_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a complete real 300-case paired export. "
            "No simulated-data fallback exists."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "analysis" / "paired_runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "analysis" / "results",
    )
    parser.add_argument("--expected-cases", type=int, default=300)
    parser.add_argument("--expected-per-cancer", type=int, default=100)
    parser.add_argument("--performance-threshold", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260621)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        data = load_and_validate_paired_data(
            args.input,
            expected_cases=args.expected_cases,
            expected_per_cancer=args.expected_per_cancer,
        )
        result = analyze_paired_data(
            data,
            input_path=args.input,
            output_dir=args.output_dir,
            performance_threshold=args.performance_threshold,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    primary = result["primary"]
    paired = result["pairedComparison"]
    print(f"Cases analyzed: {result['sample']['cases']}")
    print(
        "UNGATED PASS: "
        f"{primary['ungatedPassRate']:.2%} "
        f"(95% Wilson CI {primary['ci95LowWilson']:.2%}–"
        f"{primary['ci95HighWilson']:.2%})"
    )
    print(
        f"GATED PASS: {paired['gatedPassRate']:.2%}; "
        f"paired difference {paired['pairedRateDifference']:.2%}; "
        f"exact McNemar p={paired['mcnemarExactP']:.6g}"
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
