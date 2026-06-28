"""Command-line entry point for batch PET-CT image generation."""

import argparse
import asyncio
import sys

from config import AppConfig
from processor import run_batch_processing


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-generate PET-CT patient visualizations.")
    parser.add_argument("input", help="Input CSV or Excel file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--limit", type=int, help="Process at most N records")
    parser.add_argument("--text-cols", help="Comma-separated report column names")
    parser.add_argument("--id-col", help="Patient or examination ID column")
    parser.add_argument("--api-key", help="API key; overrides NANOBANANA_API_KEY")
    parser.add_argument("--api-url", help="OpenAI-compatible API endpoint")
    parser.add_argument("--api-mode", choices=["chat", "image"])
    parser.add_argument("--model", help="Image model name")
    parser.add_argument("--reference-image", help="Reference image path or URL")
    parser.add_argument("--size", help="Image size, for example 1024x1024")
    parser.add_argument("--merge-history", action="store_true")
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--history-file", help="CSV used to resume an interrupted batch")
    parser.add_argument("--rpm", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--timeout", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    config = AppConfig.from_env()
    if args.text_cols:
        config.csv.user_text_cols = [value.strip() for value in args.text_cols.split(",") if value.strip()]
    if args.id_col:
        config.csv.id_column_candidates = [args.id_col]
    for argument, attribute in (
        ("api_key", "api_key"),
        ("api_url", "api_url"),
        ("api_mode", "api_mode"),
        ("model", "model"),
        ("reference_image", "reference_image_url"),
        ("size", "size"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(config, attribute, value)
    for argument in ("rpm", "concurrency", "timeout"):
        value = getattr(args, argument)
        if value is not None:
            setattr(config.rate_limit, argument, value)

    if not config.api_key:
        raise SystemExit("API key missing. Set NANOBANANA_API_KEY or pass --api-key.")

    try:
        asyncio.run(
            run_batch_processing(
                args.input,
                args.output,
                config,
                limit=args.limit,
                merge_history=args.merge_history,
                random_sample=args.random_sample,
                history_file=args.history_file,
            )
        )
    except KeyboardInterrupt:
        print("Job cancelled.")
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
