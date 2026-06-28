"""Delete generated images whose source IDs appear in a table."""

import argparse
import logging
from pathlib import Path

from petct.data import detect_id_column, extract_image_record_id, read_table


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_ids_from_table(path: Path, id_col: str | None = None, no_header: bool = False) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")

    if no_header:
        import pandas as pd

        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, encoding="utf-8-sig", header=None).fillna("")
        else:
            frame = pd.read_excel(path, header=None).fillna("")
        target_col = frame.columns[0]
    else:
        frame = read_table(path)
        target_col = detect_id_column(list(frame.columns), requested=id_col)
    return {str(value).strip() for value in frame[target_col] if str(value).strip()}


def find_matching_images(image_dir: Path, target_ids: set[str]) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return [
        path
        for path in image_dir.iterdir()
        if path.is_file() and extract_image_record_id(path.name) in target_ids
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Delete generated images by source ID.")
    parser.add_argument("table", help="CSV or Excel file containing IDs")
    parser.add_argument("image_dir")
    parser.add_argument("--id-col")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    target_ids = get_ids_from_table(Path(args.table), args.id_col, args.no_header)
    matches = find_matching_images(Path(args.image_dir), target_ids)
    if not matches:
        print("No matching images found.")
        return

    action = "Would delete" if args.dry_run else "Deleting"
    print(f"{action} {len(matches)} image(s):")
    for path in matches:
        print(f"  {path.name}")
        if not args.dry_run:
            path.unlink()


if __name__ == "__main__":
    main()
