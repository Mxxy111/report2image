"""Export generated comprehension questions to a clinician review workbook."""

import argparse
from pathlib import Path

import pandas as pd

from petct.data import detect_id_column
from petct.questions import parse_generated_questions


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    default_root = PROJECT_ROOT / "data" / "derived" / "clinical_utility_dataset_full"
    parser = argparse.ArgumentParser(description="Export generated questions for clinician review.")
    parser.add_argument("--dataset-root", default=str(default_root))
    parser.add_argument("--input-csv")
    parser.add_argument("--output-excel")
    parser.add_argument("--id-col")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    input_csv = Path(args.input_csv) if args.input_csv else dataset_root / "clinical_utility_questions.csv"
    output_excel = (
        Path(args.output_excel)
        if args.output_excel
        else dataset_root / "clinical_utility_questions_review.xlsx"
    )
    if not input_csv.exists():
        raise SystemExit(f"Input file not found: {input_csv}")

    frame = pd.read_csv(input_csv, encoding="utf-8-sig").fillna("")
    id_col = detect_id_column(list(frame.columns), requested=args.id_col)
    output_rows = []
    for index, row in frame.iterrows():
        review_row = {
            "序号": index + 1,
            "病例ID": row.get(id_col, ""),
            "诊断": row.get("诊断", ""),
            "检查结论": row.get("检查结论", ""),
        }
        questions = parse_generated_questions(row.get("generated_questions"))
        for question_index in range(3):
            prefix = f"问题_{question_index + 1}"
            item = questions[question_index] if question_index < len(questions) else {}
            review_row[f"{prefix}_题目"] = item.get("question", "")
            review_row[f"{prefix}_选项"] = " | ".join(item.get("options", []))
            review_row[f"{prefix}_参考答案"] = item.get("answer", "")
            review_row[f"{prefix}_类型"] = item.get("type", "")
        output_rows.append(review_row)

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_excel(output_excel, index=False)
    print(f"Wrote {len(output_rows)} rows to {output_excel}")


if __name__ == "__main__":
    main()
