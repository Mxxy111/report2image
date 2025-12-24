import argparse
import csv
import sys
from pathlib import Path
from typing import Set, Dict, List

def scan_images(image_dir: Path) -> Set[str]:
    """扫描目录，返回所有存在图片的 ID 集合"""
    ids = set()
    if not image_dir.exists():
        return ids
    
    # 假设文件名格式: {ID}_{Hash}.png 或 {ID}.png
    for f in image_dir.glob("*.png"):
        # 提取 ID：文件名第一个下划线前的部分，或者整个文件名（去掉后缀）
        stem = f.stem
        if '_' in stem:
            img_id = stem.split('_')[0]
        else:
            img_id = stem
        ids.add(img_id)
    return ids

def load_csv_records(csv_path: Path, id_col: str = None) -> Dict[str, dict]:
    """读取 CSV，返回 {id: record_dict} 映射"""
    records = {}
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return records
    
    try:
        import pandas as pd
        if csv_path.suffix == '.csv':
            try:
                # Try UTF-8 first
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                try:
                    # Try GB18030
                    print(f"UTF-8 decode failed for {csv_path}, retrying with GB18030...")
                    df = pd.read_csv(csv_path, encoding='gb18030')
                except UnicodeDecodeError:
                     # Final attempt: decode with replacement to salvage what we can
                    print(f"GB18030 decode failed for {csv_path}. File might be mixed encoding. Attempting to recover with errors='replace'...")
                    try:
                        df = pd.read_csv(csv_path, encoding='utf-8-sig', encoding_errors='replace')
                    except Exception:
                        # If UTF-8 replace fails (rare), try GB18030 replace
                        df = pd.read_csv(csv_path, encoding='gb18030', encoding_errors='replace')
        else:
            df = pd.read_excel(csv_path)
        
        # 自动探测 ID 列
        target_col = id_col
        if not target_col:
            candidates = [c for c in df.columns if "号" in c or "ID" in c or "id" in c.lower()]
            if candidates:
                target_col = candidates[0]
        
        if not target_col:
            print(f"Error: Could not find ID column in {csv_path}")
            return {}
            
        print(f"Using ID column: {target_col}")
        
        # 转换为字典
        for _, row in df.iterrows():
            # 将所有值转为字符串，避免类型问题
            row_dict = row.to_dict()
            pid = str(row_dict[target_col]).strip()
            records[pid] = row_dict
            
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        
    return records

def main():
    parser = argparse.ArgumentParser(description="Sync History CSV with Image Directory")
    parser.add_argument("--history", required=True, help="History CSV file path")
    parser.add_argument("--images", required=True, help="Image directory path")
    parser.add_argument("--source", help="Original source CSV file path (optional, for recovering missing records)")
    parser.add_argument("--id-col", help="ID column name (optional)")
    
    args = parser.parse_args()
    
    history_path = Path(args.history)
    image_dir = Path(args.images)

    print(f"Scanning images in {image_dir}...")
    image_ids = scan_images(image_dir)
    print(f"Found {len(image_ids)} images.")

    # 如果 history 文件不存在，但提供了 source，则从图片 + 原始表反向创建一个新的 history
    if not history_path.exists():
        print(f"History file {history_path} does not exist.")
        if not args.source:
            print("No source file provided. Cannot create new history from images.")
            sys.exit(1)

        print(f"Creating new history from source: {args.source}")
        source_records = load_csv_records(Path(args.source), args.id_col)
        if not source_records:
            print("No records loaded from source file. Abort.")
            sys.exit(1)

        history_records = {}
        recovered_count = 0
        for img_id in image_ids:
            if img_id in source_records:
                history_records[img_id] = source_records[img_id]
                recovered_count += 1
            else:
                print(f"Warning: ID {img_id} not found in source file, skipping.")

        print(f"Recovered {recovered_count} records.")

        if not history_records:
            print("Result is empty, nothing to save.")
            return

        print(f"Saving new history to {history_path}...")
        try:
            import pandas as pd
            df_new = pd.DataFrame(list(history_records.values()))
            df_new.to_csv(history_path, index=False, encoding='utf-8-sig')
            print("Done.")
        except Exception as e:
            print(f"Failed to save CSV: {e}")
        return

    # 正常路径：history 已存在，执行同步逻辑
    print(f"Loading history records from {history_path}...")
    history_records = load_csv_records(history_path, args.id_col)
    print(f"Found {len(history_records)} history records.")
    
    history_ids = set(history_records.keys())
    
    # 1. 找出无效记录 (在 CSV 中但没图)
    orphans = history_ids - image_ids
    if orphans:
        print(f"\nFound {len(orphans)} orphan records (in CSV but no image). Removing them...")
        for oid in orphans:
            del history_records[oid]
    else:
        print("\nNo orphan records found.")
        
    # 2. 找出丢失记录 (有图但不在 CSV 中)
    ghosts = image_ids - history_ids
    if ghosts:
        print(f"\nFound {len(ghosts)} ghost images (image exists but not in CSV).")
        if args.source:
            print(f"Attempting to recover from source: {args.source}")
            source_records = load_csv_records(Path(args.source), args.id_col)
            
            recovered_count = 0
            for gid in ghosts:
                if gid in source_records:
                    history_records[gid] = source_records[gid]
                    recovered_count += 1
                else:
                    print(f"Warning: ID {gid} not found in source file, cannot recover.")
            print(f"Recovered {recovered_count} records.")
        else:
            print("No source file provided. These records will remain missing from CSV.")
            print("Use --source to provide the original dataset for recovery.")
    else:
        print("\nNo ghost images found.")
        
    # 3. 保存更新后的 CSV
    if orphans or (ghosts and args.source):
        backup_path = history_path.with_suffix(history_path.suffix + ".bak")
        print(f"\nBacking up original history to {backup_path}")
        import shutil
        shutil.copy2(history_path, backup_path)
        
        print(f"Saving synced history to {history_path}...")
        try:
            # 获取所有字段名
            if not history_records:
                print("Result is empty, skipping save.")
                return

            import pandas as pd
            df_new = pd.DataFrame(list(history_records.values()))
            df_new.to_csv(history_path, index=False, encoding='utf-8-sig')
            print("Done.")
            
        except Exception as e:
            print(f"Failed to save CSV: {e}")
    else:
        print("\nHistory is already in sync.")

if __name__ == "__main__":
    main()
