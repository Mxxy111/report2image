"""Export a paired experiment into the strict statistical-analysis table."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from petct.experiment import export_experiment_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "runtime" / "experiments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "analysis" / "paired_runs.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_root / args.experiment_id
    try:
        rows = export_experiment_results(
            experiment_dir=experiment_dir,
            project_root=PROJECT_ROOT,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    statuses = Counter(str(row["run_status"]) for row in rows)
    decisions = Counter(
        str(row["human_decision"])
        for row in rows
        if row["human_decision"]
    )
    print(f"Exported {len(rows)} planned runs to {args.output}")
    print(f"Run status: {dict(statuses)}")
    print(f"Human decisions: {dict(decisions)}")
    if statuses.get("MISSING") or sum(decisions.values()) != len(rows):
        print(
            "Dataset is not yet ready for formal analysis: every planned run must "
            "be completed and receive a human PASS/FAIL decision."
        )


if __name__ == "__main__":
    main()
