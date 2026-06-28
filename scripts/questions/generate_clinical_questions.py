"""Generate patient-comprehension questions from PET-CT reports."""

import argparse
import concurrent.futures
import csv
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from petct.data import compose_report_text, detect_text_columns
from petct.questions import parse_generated_questions


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """你是一名核医学科医生。请基于 PET-CT 报告生成 3 道患者理解度测试题。

要求：
1. 题目必须能直接从报告中作答，不得引入报告外推断。
2. 优先覆盖病灶位置、良恶性倾向、转移情况、病灶大小或 SUVmax。
3. 使用患者能理解的中文，避免考查专业术语记忆。
4. 每题为单选题，提供 4 个互斥选项，只有 1 个正确答案。
5. 左右侧、器官和数值必须与原报告完全一致。

仅输出 JSON：
{
  "questions": [
    {
      "question": "问题",
      "options": ["A", "B", "C", "D"],
      "answer": "与某个选项完全一致的正确答案",
      "type": "定位/存在性/定量/性质"
    }
  ]
}
"""


def generate_questions(client, row, model_name, text_columns, max_retries=3):
    report_text = compose_report_text(row, text_columns)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请根据以下报告生成 3 道题：\n\n{report_text}"},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1600,
                response_format={"type": "json_object"},
            )
            questions = parse_generated_questions(response.choices[0].message.content)
            if questions:
                return json.dumps(questions, ensure_ascii=False)
            print(f"[WARN] Invalid question JSON for {row.get('image_filename', 'unknown')}")
        except Exception as exc:
            print(f"[WARN] Attempt {attempt + 1} failed for {row.get('image_filename', 'unknown')}: {exc}")
            if attempt + 1 < max_retries:
                time.sleep(2**attempt)
    return "[]"


def parse_args():
    default_root = PROJECT_ROOT / "data" / "derived" / "clinical_utility_dataset_full"
    parser = argparse.ArgumentParser(
        description="Generate patient-comprehension questions from PET-CT metadata."
    )
    parser.add_argument("--dataset-root", default=str(default_root))
    parser.add_argument("--input-csv")
    parser.add_argument("--output-csv")
    parser.add_argument("--text-cols", help="Comma-separated report column names")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Environment variable {args.api_key_env} is not set.")

    dataset_root = Path(args.dataset_root)
    input_csv = Path(args.input_csv) if args.input_csv else dataset_root / "clinical_utility_metadata_full.csv"
    output_csv = Path(args.output_csv) if args.output_csv else dataset_root / "clinical_utility_questions.csv"
    if not input_csv.exists():
        raise SystemExit(f"Input file not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    requested = [value.strip() for value in args.text_cols.split(",")] if args.text_cols else None
    text_columns = detect_text_columns(fieldnames, requested=requested)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    output_rows = [row.copy() for row in rows]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(
                generate_questions,
                client,
                row,
                args.model,
                text_columns,
                args.max_retries,
            ): index
            for index, row in enumerate(rows)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            index = futures[future]
            try:
                output_rows[index]["generated_questions"] = future.result()
            except Exception as exc:
                print(f"[WARN] Unhandled error for row {index}: {exc}")
                output_rows[index]["generated_questions"] = "[]"

    if "generated_questions" not in fieldnames:
        fieldnames.append("generated_questions")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
