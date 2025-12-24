import argparse
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def read_csv_robust(path: Path, header=0):
    """Read CSV with encoding fallback (UTF-8 -> GB18030)."""
    import pandas as pd
    try:
        return pd.read_csv(path, encoding='utf-8-sig', header=header)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding='gb18030', header=header)
        except UnicodeDecodeError:
            # Final attempt: decode with replacement
            return pd.read_csv(path, encoding='utf-8-sig', encoding_errors='replace', header=header)

def get_ids_from_csv(csv_path: Path, id_col: str = None, no_header: bool = False) -> set:
    """Read IDs from CSV file."""
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return set()
    
    try:
        import pandas as pd
        header_arg = None if no_header else 0
        
        if csv_path.suffix.lower() == '.csv':
            df = read_csv_robust(csv_path, header=header_arg)
        else:
            df = pd.read_excel(csv_path, header=header_arg)
            
        # Auto-detect ID column if not provided
        target_col = id_col
        if no_header:
            # Use the first column
            target_col = df.columns[0]
        elif not target_col:
            candidates = [c for c in df.columns if "号" in c or "ID" in c or "id" in c.lower()]
            if candidates:
                target_col = candidates[0]
        
        if target_col is None: # explicit check for None/empty
            logger.error(f"Could not find ID column in {csv_path}. Please specify with --id-col")
            logger.info(f"Available columns: {list(df.columns)}")
            return set()
            
        logger.info(f"Using ID column: {target_col}")
        
        # Extract IDs as strings and strip whitespace
        ids = set(df[target_col].astype(str).str.strip().values)
        return ids
        
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return set()

def delete_images(image_dir: Path, target_ids: set, dry_run: bool = False):
    """Delete images matching the target IDs."""
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return

    logger.info(f"Scanning {image_dir} for {len(target_ids)} target IDs...")
    
    deleted_count = 0
    found_files = []

    # Iterate over all files in the directory
    for file_path in image_dir.glob("*"):
        if not file_path.is_file():
            continue
            
        # Check if filename starts with any of the target IDs
        # Filename format expected: {ID}_{Hash}.png or {ID}.png
        stem = file_path.stem
        
        # Extract ID part (everything before first underscore, or whole stem)
        if '_' in stem:
            file_id = stem.split('_')[0]
        else:
            file_id = stem
            
        if file_id in target_ids:
            found_files.append(file_path)

    if not found_files:
        logger.info("No matching images found.")
        return

    logger.info(f"Found {len(found_files)} images matching the provided IDs.")
    
    if dry_run:
        logger.info("Dry run enabled. The following files WOULD be deleted:")
        for f in found_files:
            logger.info(f"  - {f.name}")
    else:
        for f in found_files:
            try:
                os.remove(f)
                logger.info(f"Deleted: {f.name}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {f.name}: {e}")
        
        logger.info(f"Successfully deleted {deleted_count} images.")

def main():
    parser = argparse.ArgumentParser(description="Delete images based on IDs from a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file containing IDs to delete")
    parser.add_argument("image_dir", help="Path to the directory containing images")
    parser.add_argument("--id-col", help="Name of the ID column in CSV (optional)")
    parser.add_argument("--no-header", action="store_true", help="CSV has no header row")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    image_dir = Path(args.image_dir)
    
    target_ids = get_ids_from_csv(csv_path, args.id_col, args.no_header)
    
    if target_ids:
        logger.info(f"Loaded {len(target_ids)} IDs from CSV.")
        delete_images(image_dir, target_ids, args.dry_run)
    else:
        logger.info("No IDs found to process.")

if __name__ == "__main__":
    main()
