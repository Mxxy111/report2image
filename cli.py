"""CLI for NanoBanana PET-CT Generator."""

import argparse
import asyncio
import sys
from pathlib import Path

from config import AppConfig
from processor import run_batch_processing

def main():
    parser = argparse.ArgumentParser(description="Batch Generate PET-CT Visualizations using NanoBanana API")
    
    parser.add_argument("input", help="Input CSV/Excel file path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--limit", type=int, help="Only process the first N records (useful for testing)")
    
    # 数据配置
    parser.add_argument("--text-cols", help="指定作为报告内容的列名，用逗号分隔 (例如: '检查所见,结论')")
    parser.add_argument("--id-col", help="指定作为ID的列名")

    parser.add_argument("--api-key", help="NanoBanana API Key (overrides env var)")
    parser.add_argument("--api-url", help="NanoBanana API URL")
    parser.add_argument("--api-mode", choices=["chat", "image"], help="API Mode: chat (default) or image")
    parser.add_argument("--model", help="Model name (overrides env var)")
    parser.add_argument("--reference-image", help="Optional reference image URL for img2img")
    parser.add_argument("--size", help="Image size (e.g. 1024x1024), only for image mode")
    
    parser.add_argument("--merge-history", action="store_true", help="当检测到'前片/较前'等字样时，自动合并同患者的历史报告")
    
    parser.add_argument("--random-sample", action="store_true", help="随机抽样模式：每个患者ID只随机取一条记录，然后随机打乱，再应用 limit")
    
    parser.add_argument("--history-file", help="指定历史记录文件（CSV）。用于【正式模式】：自动排除该文件中已存在的ID，并将新处理的记录追加到该文件中。")

    parser.add_argument("--rpm", type=int, help="Requests per minute limit")
    parser.add_argument("--concurrency", type=int, help="Max concurrent requests")
    parser.add_argument("--timeout", type=float, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Update Config
    config = AppConfig.from_env()
    
    if args.text_cols:
        config.csv.user_text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    if args.id_col:
        config.csv.id_column_candidates = [args.id_col] # 优先使用指定的 ID 列

    if args.api_key:
        config.api_key = args.api_key
    if args.api_url:
        config.api_url = args.api_url
    if args.api_mode:
        config.api_mode = args.api_mode
    if args.model:
        config.model = args.model
    if args.reference_image:
        config.reference_image_url = args.reference_image
    if args.size:
        config.size = args.size
    if args.rpm:
        config.rate_limit.rpm = args.rpm
    if args.concurrency:
        config.rate_limit.concurrency = args.concurrency
    if args.timeout:
        config.rate_limit.timeout = args.timeout

    if not config.api_key:
        print("Error: API Key not found. Please provide --api-key or set NANOBANANA_API_KEY env var.")
        sys.exit(1)

    print(f"Starting Job: Input={args.input}")
    
    try:
        asyncio.run(run_batch_processing(
            args.input, 
            args.output, 
            config, 
            limit=args.limit,
            merge_history=args.merge_history,
            random_sample=args.random_sample,
            history_file=args.history_file
        ))
    except KeyboardInterrupt:
        print("\nJob cancelled by user.")
    except Exception as e:
        print(f"\nFatal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
