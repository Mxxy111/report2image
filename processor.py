"""Core Batch Processor for Image Generation."""

import asyncio
import csv
import hashlib
import time
import logging
import os
import aiohttp
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from config import AppConfig
from client import NanoBananaClient
from prompts import PET_CT_VISUALIZATION_PROMPT, PET_CT_IMG2IMG_PROMPT

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BatchStats:
    total_api_calls: int = 0
    success_api_calls: int = 0
    failed_api_calls: int = 0

@dataclass
class Record:
    id_value: str
    text: str
    row_data: Dict[str, Any]
    task_hash: str

@dataclass
class ProcessingResult:
    id_value: str
    original_text: str
    image_url: Optional[str] = None
    local_image_path: Optional[str] = None
    error: Optional[str] = None
    elapsed_time: float = 0.0

class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self._lock = asyncio.Lock()
        self._next_slot = time.monotonic()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._next_slot:
                await asyncio.sleep(self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self.interval

def read_csv_robust(path: Path, default_encoding: str = "utf-8-sig") -> Any:
    """Read CSV with encoding fallback (UTF-8 -> GB18030)."""
    import pandas as pd
    try:
        return pd.read_csv(path, encoding=default_encoding)
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {path}, retrying with GB18030...")
        return pd.read_csv(path, encoding='gb18030')

def read_records(path: Path, config: AppConfig, limit: Optional[int] = None) -> Tuple[List[Record], str]:
    """Read CSV records."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    import pandas as pd
    
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path).fillna("")
    else:
        df = read_csv_robust(path, config.csv.default_encoding).fillna("")
    
    columns = list(df.columns)
    
    # Detect Text Column(s)
    text_columns = []
    if config.csv.user_text_cols:
        # 用户指定了列名
        for user_col in config.csv.user_text_cols:
            if user_col in columns:
                text_columns.append(user_col)
            else:
                logger.warning(f"Specified text column '{user_col}' not found in CSV. Available: {columns}")
        
        if not text_columns:
            raise ValueError(f"None of the specified text columns found in CSV: {config.csv.user_text_cols}")
    else:
        # 自动检测
        detected_col = next((c for c in columns if c in config.csv.text_column_candidates), None)
        if not detected_col:
            # Try fuzzy match
            detected_col = next((c for c in columns if "检查所见" in c or "report" in c.lower()), columns[0])
            logger.warning(f"Text column not explicitly found, using: {detected_col}")
        text_columns = [detected_col]

    # Detect ID Column
    id_col = next((c for c in columns if c in config.csv.id_column_candidates), None)
    if not id_col:
        id_col = next((c for c in columns if "号" in c or "ID" in c or "id" in c.lower()), columns[0])
        logger.warning(f"ID column not explicitly found, using: {id_col}")

    records = []
    for _, row in df.iterrows():
        # Concatenate text from multiple columns
        text_parts = []
        for col in text_columns:
            val = str(row.get(col, "")).strip()
            if val:
                # 如果是多列，可以加上列名作为小标题 (可选，这里直接拼内容)
                # text_parts.append(f"【{col}】\n{val}") 
                text_parts.append(val)
        
        text = "\n\n".join(text_parts)
        
        if not text:
            continue
            
        id_val = str(row.get(id_col, "")).strip()
        if not id_val:
            id_val = f"row_{len(records)}"
            
        # Generate hash
        task_hash = hashlib.md5(f"{id_val}_{text}".encode()).hexdigest()
        
        records.append(Record(
            id_value=id_val,
            text=text,
            row_data=row.to_dict(),
            task_hash=task_hash
        ))
        
        if limit and len(records) >= limit:
            break
        
    return records, id_col

async def download_image(session: aiohttp.ClientSession, url: str, save_path: Path) -> bool:
    """Download image from URL to local path."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
                return True
            else:
                logger.error(f"Failed to download image from {url}: {response.status}")
                return False
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return False

import re

def extract_image_url(content: str) -> Optional[str]:
    """
    从 Chat 文本中提取图片 URL 或 Base64。
    支持 markdown 格式 ![...](url) 或直接 http 链接。
    """
    # 0. Check for Markdown image with Base64
    # match: ![image](data:image/jpeg;base64,...)
    b64_match = re.search(r'!\[.*?\]\((data:image/[^;]+;base64,[^\)]+)\)', content)
    if b64_match:
        return b64_match.group(1)

    # 1. Markdown image regex (HTTP)
    md_match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', content)
    if md_match:
        return md_match.group(1)
    
    # 2. Direct URL regex (png/jpg/webp)
    url_match = re.search(r'(https?://[^\s]+\.(?:png|jpg|jpeg|webp))', content, re.IGNORECASE)
    if url_match:
        return url_match.group(1)
        
    # 3. Fallback: just find any http url (might capture non-image urls)
    any_url = re.search(r'(https?://[^\s\)]+)', content)
    if any_url:
        return any_url.group(1)
        
    return None

async def save_base64_image(data_uri: str, save_path: Path) -> bool:
    """Decode and save Base64 image."""
    try:
        header, encoded = data_uri.split(",", 1)
        data = base64.b64decode(encoded)
        with open(save_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        logger.error(f"Error saving base64 image: {e}")
        return False

import json

def log_api_call(output_dir: Path, record_id: str, prompt: str, model: str, success: bool, response: Any = None, error: str = None):
    """
    记录详细的 API 调用日志到 JSONL 文件
    """
    log_dir = output_dir / "call_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"api_calls_{time.strftime('%Y%m%d')}.jsonl"
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "record_id": str(record_id),
        "model": model,
        "prompt_length": len(prompt),
        "success": success,
        "error": str(error) if error else None,
        "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt, # 预览前500字符
        # "full_prompt": prompt, # 如果需要完整prompt用于调试，可以取消注释，但文件会很大
    }
    
    if response and isinstance(response, dict):
        # 尝试提取 token usage
        if "usage" in response:
            log_entry["usage"] = response["usage"]
        
        # 提取返回内容片段
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            log_entry["response_preview"] = content[:200] + "..." if len(content) > 200 else content

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write API log: {e}")

async def process_single_record(
    record: Record,
    client: NanoBananaClient,
    config: AppConfig,
    limiter: RateLimiter,
    sem: asyncio.Semaphore,
    image_output_dir: Path,
    stats: BatchStats
) -> ProcessingResult:
    start_time = time.monotonic()
    
    # Select prompt based on mode
    if config.reference_image_url:
        # 垫图模式使用专用提示词
        prompt_template = PET_CT_IMG2IMG_PROMPT
    else:
        # 纯文本模式使用标准提示词
        prompt_template = PET_CT_VISUALIZATION_PROMPT
        
    prompt = prompt_template.format(report_content=record.text)
    
    # Determine log directory (parent of images dir)
    log_output_dir = image_output_dir.parent.parent
    
    # Retry logic
    last_error = None
    for attempt in range(config.rate_limit.max_retries + 1):
        try:
            async with sem:
                await limiter.wait()
                # Increment total calls (atomic in GIL)
                stats.total_api_calls += 1
                response = await client.generate_image(prompt)
            
            # Log successful call
            stats.success_api_calls += 1
            log_api_call(log_output_dir, record.id_value, prompt, config.model, True, response=response)

            # Parse response for Chat Completion
            # Format: choices[0].message.content
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                
                # Extract Image URL from content
                image_source = extract_image_url(content)
                
                if not image_source:
                     # 打印出模型返回的具体文本，方便排查拒单原因（如安全过滤）
                     preview = content[:200].replace('\n', ' ')
                     raise Exception(f"No image URL found. Response content: '{preview}...'")

                # Download image
                file_name = f"{record.id_value}_{record.task_hash[:6]}.png"
                # Sanitize filename
                file_name = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c in ('-', '_', '.')])
                local_path = image_output_dir / file_name
                
                success = False
                if image_source.startswith("data:"):
                    # Handle Base64
                    success = await save_base64_image(image_source, local_path)
                    image_url_log = "base64_image"
                else:
                    # Handle URL
                    async with aiohttp.ClientSession() as session:
                        success = await download_image(session, image_source, local_path)
                    image_url_log = image_source
                
                if success:
                    return ProcessingResult(
                        id_value=record.id_value,
                        original_text=record.text,
                        image_url=image_url_log,
                        local_image_path=str(local_path),
                        elapsed_time=time.monotonic() - start_time
                    )
                else:
                    raise Exception("Failed to save/download image")
            else:
                raise Exception(f"Unexpected response format: {response}")

        except Exception as e:
            last_error = str(e) or f"Error: {type(e).__name__}"
            # Log failed attempt
            stats.failed_api_calls += 1
            log_api_call(log_output_dir, record.id_value, prompt, config.model, False, error=str(e))
            
            if attempt < config.rate_limit.max_retries:
                delay = config.rate_limit.retry_backoff * (2 ** attempt)
                logger.warning(f"Attempt {attempt+1} failed for {record.id_value}: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All attempts failed for {record.id_value}: {e}")
                
    return ProcessingResult(
        id_value=record.id_value,
        original_text=record.text,
        error=last_error,
        elapsed_time=time.monotonic() - start_time
    )

def enrich_records_with_history(records: List[Record], config: AppConfig) -> List[Record]:
    """
    根据 ID 和日期对记录进行分组排序。
    若检测到关键词，合并历史报告内容。
    """
    if not records:
        return records

    import pandas as pd
    
    logger.info("Processing history merging...")
    
    # 1. Group by ID
    grouped = {}
    for r in records:
        if r.id_value not in grouped:
            grouped[r.id_value] = []
        grouped[r.id_value].append(r)

    keywords = ["前片", "较前", "同前", "复查", "相仿", "对比", "20"] 
    # 增加 "20" 是为了匹配日期引用如 "对比2022.12.07日片"

    processed_records = []
    
    # 2. Process each group
    for pid, group in grouped.items():
        # Try to find date column in row_data
        # Use the first key that looks like a date or contains 'time'/'date'/'日期'
        if not group:
            continue
            
        sample_row = group[0].row_data
        date_col = None
        
        # Heuristic to find date column
        candidates = [k for k in sample_row.keys() if any(x in k.lower() for x in ['date', 'time', '日期', '时间'])]
        if candidates:
            date_col = candidates[0]
        else:
            # Fallback: try to parse the 4th column if exists (common in this dataset)
            keys = list(sample_row.keys())
            if len(keys) > 3:
                date_col = keys[3]
        
        if date_col:
            # Sort by date
            try:
                group.sort(key=lambda x: pd.to_datetime(x.row_data.get(date_col, ""), errors='coerce') or pd.Timestamp.min)
            except Exception:
                pass # Sort failed, keep original order
        
        # 3. Merge text
        history_texts = []
        
        for i, record in enumerate(group):
            current_text = record.text
            
            # Check for keywords
            needs_merge = any(k in current_text for k in keywords)
            
            if needs_merge and i > 0:
                # Construct history context
                # Limit history to last 3 reports to avoid token overflow
                relevant_history = history_texts[-3:] 
                
                merged_context = ""
                for h_date, h_text in relevant_history:
                    merged_context += f"【历史报告 {h_date}】\n{h_text}\n\n"
                
                if merged_context:
                    # Prepend history to current text
                    # Add a separator for clarity
                    new_text = f"{merged_context}【本次报告 {record.row_data.get(date_col, '本次')}】\n{current_text}"
                    
                    # Update record text (create new Record to be safe)
                    record.text = new_text
            
            # Add current (original) text to history for future records
            # Use date if available
            date_str = str(record.row_data.get(date_col, f"Report-{i+1}"))
            history_texts.append((date_str, current_text))
            
            processed_records.append(record)

    return processed_records

import random

def apply_random_sampling(records: List[Record], limit: int) -> List[Record]:
    """
    随机抽样：每个ID只保留一条记录，然后随机打乱，取前 limit 条。
    """
    if not records:
        return []
        
    logger.info("Applying random sampling (unique ID per sample)...")
    
    # Group by ID
    grouped = {}
    for r in records:
        if r.id_value not in grouped:
            grouped[r.id_value] = []
        grouped[r.id_value].append(r)
        
    # Randomly select one from each group
    sampled_records = []
    for pid, group in grouped.items():
        sampled_records.append(random.choice(group))
        
    # Shuffle
    random.shuffle(sampled_records)
    
    # Apply limit
    if limit and limit < len(sampled_records):
        logger.info(f"Sampling {limit} unique patients from {len(sampled_records)} available.")
        return sampled_records[:limit]
    
    return sampled_records

def load_processed_ids(history_file: str, id_col_name: str) -> set:
    """读取历史文件中的 ID 列表"""
    history_path = Path(history_file)
    processed_ids = set()
    
    if not history_path.exists():
        return processed_ids
        
    try:
        import pandas as pd
        # 尝试使用 pandas 读取，因为它更健壮
        if history_path.suffix == '.csv':
            df = read_csv_robust(history_path)
        else:
            df = pd.read_excel(history_path)
        
        # 尝试找 ID 列
        target_col = None
        if id_col_name in df.columns:
            target_col = id_col_name
        else:
             # 尝试模糊匹配
             candidates = [c for c in df.columns if "号" in c or "ID" in c or "id" in c.lower()]
             if candidates:
                 target_col = candidates[0]
        
        if target_col:
            processed_ids = set(df[target_col].astype(str).unique())
            logger.info(f"Loaded {len(processed_ids)} processed IDs from {history_file}")
        else:
            logger.warning(f"Could not find ID column '{id_col_name}' in history file.")
            
    except Exception as e:
        logger.warning(f"Failed to load history file: {e}")
        
    return processed_ids

def save_batch_original_csv(records: List[Record], output_path: Path, fieldnames: List[str] = None):
    """保存当前批次的原始数据"""
    if not records:
        return

    # 获取所有可能的字段名
    if not fieldnames:
        fieldnames = list(records[0].row_data.keys())
        
    try:
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                # 确保写入的是原始 row_data
                # 注意：有些字段可能缺失，DictWriter 会处理
                writer.writerow(r.row_data)
        logger.info(f"Saved batch original data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save batch CSV: {e}")

def append_to_history(records: List[Record], history_file: str):
    """追加记录到历史文件"""
    history_path = Path(history_file)
    file_exists = history_path.exists()
    
    if not records:
        return

    fieldnames = list(records[0].row_data.keys())
    
    # 默认编码
    encoding = 'utf-8-sig'
    
    # 如果文件存在，检测其编码
    if file_exists:
        try:
            with open(history_path, 'r', encoding='utf-8-sig') as f:
                f.read(1024)
        except UnicodeDecodeError:
            # UTF-8 失败，尝试 GB18030
            try:
                with open(history_path, 'r', encoding='gb18030') as f:
                    f.read(1024)
                encoding = 'gb18030'
                logger.info(f"Detected existing history file encoding: {encoding}")
            except Exception:
                pass # 保持默认

    try:
        # 如果文件不存在，写入头；如果存在，追加
        mode = 'a' if file_exists else 'w'
        with open(history_path, mode, encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in records:
                writer.writerow(r.row_data)
        logger.info(f"Appended {len(records)} records to {history_file}")
    except Exception as e:
        logger.error(f"Failed to append to history file: {e}")

async def run_batch_processing(
    input_file: str,
    output_dir: str,
    config: AppConfig = None,
    limit: Optional[int] = None,
    merge_history: bool = False,
    random_sample: bool = False,
    history_file: str = None
):
    if config is None:
        config = AppConfig.from_env()

    input_path = Path(input_file)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Determine directory structure
    if history_file:
        # 正式模式：基于历史文件名创建固定任务目录
        task_name = Path(history_file).stem
        task_root = out_path / task_name
        
        # 图片统一存放
        images_dir = task_root / "images"
        
        # 本次批次的日志/CSV 单独存放
        session_dir = task_root / "batches" / f"batch_{timestamp}"
    else:
        # 默认模式：每次都是独立的 batch 目录
        session_dir = out_path / f"batch_{timestamp}"
        images_dir = session_dir / "images"

    images_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading records from {input_path}...")
    try:
        # 决定是否需要读取所有记录
        # 如果开启了随机抽样 OR 历史合并 OR 历史排重，都需要先读取全部数据以便后续处理
        need_all_records = random_sample or merge_history or (history_file is not None)
        read_limit = None if need_all_records else limit
        
        records, id_col_name = read_records(input_path, config, limit=read_limit)
        
        # 0. 历史排重 (Incremental Mode)
        if history_file:
            processed_ids = load_processed_ids(history_file, id_col_name)
            if processed_ids:
                original_count = len(records)
                records = [r for r in records if str(r.id_value) not in processed_ids]
                filtered_count = len(records)
                logger.info(f"Filtered out {original_count - filtered_count} already processed IDs.")
        
        # 1. 历史合并 (依赖完整时间线)
        if merge_history:
            logger.info("History merging enabled. Reading all records first...")
            records = enrich_records_with_history(records, config)
            
        # 2. 随机抽样 (依赖去重)
        if random_sample:
            if limit:
                records = apply_random_sampling(records, limit)
            else:
                logger.warning("Random sample enabled but no limit specified. Returning all unique patients shuffled.")
                records = apply_random_sampling(records, len(records))
        
        # 3. 普通 Limit (如果前面没处理过 limit)
        elif limit and not merge_history: 
            # 如果 merge_history=True，limit 已经在上面 read_limit=None 时失效了，需要在最后应用
            # 但如果在 enrich 之后应用切片操作 `records[:limit]`，只会截取前N个（通常是同一个人的），
            # 所以对于 merge_history=True 但 random_sample=False 的情况，我们在这里手动切片
            # 修正逻辑：如果设置了 limit 且没走 random_sample，这里要截断
            if len(records) > limit:
                 records = records[:limit]
            
        if limit and merge_history and not random_sample:
             records = records[:limit]
             
        # === 保存本次选中的原始数据 (Batch Original CSV) ===
        if records:
            batch_csv_name = f"selected_records_{timestamp}.csv"
            save_batch_original_csv(records, session_dir / batch_csv_name)
            logger.info(f"Selected {len(records)} records for processing.")
                
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return

    if not records:
        logger.warning("No records found to process.")
        return

    logger.info(f"Starting batch processing for {len(records)} records.")
    logger.info(f"Concurrency: {config.rate_limit.concurrency}, RPM: {config.rate_limit.rpm}")

    client = NanoBananaClient(config)
    limiter = RateLimiter(config.rate_limit.rpm)
    sem = asyncio.Semaphore(config.rate_limit.concurrency)
    
    stats = BatchStats()

    tasks = [
        process_single_record(r, client, config, limiter, sem, images_dir, stats)
        for r in records
    ]

    results = []
    # Use tqdm for progress bar if available, otherwise just gather
    try:
        from tqdm.asyncio import tqdm
        for f in tqdm.as_completed(tasks, total=len(tasks), desc="Generating Images"):
            results.append(await f)
    except ImportError:
        results = await asyncio.gather(*tasks)

    # Save summary CSV
    csv_path = session_dir / "results_summary.csv"
    # 新增 original_text 列，方便用户对照
    fieldnames = [id_col_name, "original_text", "local_image_path", "image_url", "error", "elapsed_time"]
    
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                id_col_name: r.id_value,
                "original_text": r.original_text, # 写入原文
                "local_image_path": r.local_image_path or "",
                "image_url": r.image_url or "",
                "error": r.error or "",
                "elapsed_time": f"{r.elapsed_time:.2f}"
            })

    logger.info(f"Batch processing complete. Results saved to {csv_path}")
    
    logger.info("=== API Usage Stats ===")
    logger.info(f"Total API Calls: {stats.total_api_calls}")
    logger.info(f" - Success: {stats.success_api_calls}")
    logger.info(f" - Failed: {stats.failed_api_calls}")
    logger.info("=======================")
    
    success_count = sum(1 for r in results if not r.error)
    failed_results = [r for r in results if r.error]
    
    # === 更新历史文件 ===
    if history_file:
        # 找出成功的记录对象
        success_ids = set(r.id_value for r in results if not r.error)
        
        if success_ids:
            success_records = [r for r in records if r.id_value in success_ids]
            append_to_history(success_records, history_file)
            logger.info(f"Successfully appended {len(success_records)} records to history.")
            
        skipped_count = len(records) - len(success_ids)
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} failed records (not added to history).")
    
    logger.info(f"Success rate: {success_count}/{len(records)}")
    
    if failed_results:
        logger.warning(f"Total Failures: {len(failed_results)}")
        logger.warning("Failed IDs:")
        for r in failed_results:
            logger.warning(f" - ID: {r.id_value}, Error: {r.error}")

if __name__ == "__main__":
    # Simple test run if executed directly
    import sys
    if len(sys.argv) > 1:
        asyncio.run(run_batch_processing(sys.argv[1], "outputs"))
    else:
        print("Usage: python processor.py <input_csv_file>")
