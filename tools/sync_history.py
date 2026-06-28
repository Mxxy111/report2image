"""Synchronize a history table with generated image files."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from petct.data import detect_id_column, extract_image_record_id, read_table


def scan_image_ids(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return {
        extract_image_record_id(path.name)
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    }


def load_records(path: Path, id_col: str | None = None) -> tuple[str, dict[str, dict]]:
    frame = read_table(path)
    target_col = detect_id_column(list(frame.columns), requested=id_col)
    records = {
        str(row[target_col]).strip(): row.to_dict()
        for _, row in frame.iterrows()
        if str(row[target_col]).strip()
    }
    return target_col, records


def write_records(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(records.values())).to_csv(path, index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser(description="Synchronize history CSV with generated images.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--source", help="Original source table used to recover missing history rows")
    parser.add_argument("--id-col")
    return parser.parse_args()


def main():
    args = parse_args()
    history_path = Path(args.history)
    image_ids = scan_image_ids(Path(args.images))
    source_path = Path(args.source) if args.source else None

    if not history_path.exists():
        if not source_path:
            raise SystemExit("History does not exist; --source is required to rebuild it.")
        _, source_records = load_records(source_path, args.id_col)
        recovered = {key: source_records[key] for key in image_ids if key in source_records}
        write_records(history_path, recovered)
        print(f"Created {history_path} with {len(recovered)} row(s).")
        return

    _, history_records = load_records(history_path, args.id_col)
    history_ids = set(history_records)
    orphan_ids = history_ids - image_ids
    for key in orphan_ids:
        history_records.pop(key)

    recovered_count = 0
    ghost_ids = image_ids - history_ids
    if ghost_ids and source_path:
        _, source_records = load_records(source_path, args.id_col)
        for key in ghost_ids:
            if key in source_records:
                history_records[key] = source_records[key]
                recovered_count += 1

    if orphan_ids or recovered_count:
        backup_path = history_path.with_suffix(history_path.suffix + ".bak")
        shutil.copy2(history_path, backup_path)
        write_records(history_path, history_records)
        print(
            f"Updated {history_path}: removed {len(orphan_ids)}, "
            f"recovered {recovered_count}; backup: {backup_path}"
        )
    else:
        print(
            f"History already synchronized. "
            f"{len(ghost_ids)} image(s) have no history row."
        )


if __name__ == "__main__":
    main()
