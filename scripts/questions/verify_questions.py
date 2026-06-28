"""Validate generated question JSON in a metadata CSV."""

import argparse
import csv
import json
from pathlib import Path

from petct.questions import parse_generated_questions


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate generated PET-CT comprehension questions.")
    parser.add_argument(
        "--input-csv",
        default=str(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "clinical_utility_dataset_full"
            / "clinical_utility_questions.csv"
        ),
    )
    parser.add_argument("--expected-count", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise SystemExit(f"Input file not found: {csv_path}")

    total = valid = invalid = 0
    example = None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            questions = parse_generated_questions(row.get("generated_questions"))
            if len(questions) == args.expected_count:
                valid += 1
                example = example or questions
            else:
                invalid += 1
                label = row.get("image_filename") or row.get("ID") or f"row {total}"
                print(f"[INVALID] {label}: expected {args.expected_count}, got {len(questions)}")

    print(f"Total rows: {total}")
    print(f"Valid question sets: {valid}")
    print(f"Invalid question sets: {invalid}")
    if example:
        print("\nExample:")
        print(json.dumps(example, indent=2, ensure_ascii=False))
    raise SystemExit(0 if invalid == 0 else 1)


if __name__ == "__main__":
    main()
