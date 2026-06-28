"""Shared data and filename helpers for PET-CT report workflows."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_ID_COLUMNS = (
    "门诊号/住院号",
    "门诊号",
    "住院号",
    "ID",
    "id",
    "patient_id",
    "accession_number",
)
DEFAULT_TEXT_COLUMNS = ("检查所见", "检查结论", "诊断", "报告内容", "report", "content")
HASH_SUFFIX_RE = re.compile(r"^[0-9a-fA-F]{6,32}$")


def read_table(path: str | Path, default_encoding: str = "utf-8-sig") -> pd.DataFrame:
    """Read a CSV or Excel file with the project's common encoding fallback."""
    table_path = Path(path)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(table_path).fillna("")

    try:
        return pd.read_csv(table_path, encoding=default_encoding).fillna("")
    except UnicodeDecodeError:
        return pd.read_csv(table_path, encoding="gb18030").fillna("")


def detect_id_column(
    columns: Sequence[str],
    requested: str | None = None,
    candidates: Iterable[str] = DEFAULT_ID_COLUMNS,
) -> str:
    """Return the best ID column name or raise a clear error."""
    if requested:
        if requested in columns:
            return requested
        raise ValueError(f"ID column not found: {requested}")

    for candidate in candidates:
        if candidate in columns:
            return candidate

    for column in columns:
        lowered = column.lower()
        if "号" in column or "id" in lowered:
            return column

    raise ValueError(f"Could not detect an ID column from: {list(columns)}")


def detect_text_columns(
    columns: Sequence[str],
    requested: Sequence[str] | None = None,
    candidates: Iterable[str] = DEFAULT_TEXT_COLUMNS,
) -> list[str]:
    """Return report text columns in the requested or detected order."""
    if requested:
        missing = [column for column in requested if column not in columns]
        if missing:
            raise ValueError(f"Text column(s) not found: {missing}")
        return list(requested)

    detected = [candidate for candidate in candidates if candidate in columns]
    if detected:
        return detected

    for column in columns:
        lowered = column.lower()
        if "检查所见" in column or "检查结论" in column or "报告" in column or "report" in lowered:
            return [column]

    raise ValueError(f"Could not detect report text columns from: {list(columns)}")


def compose_report_text(row: dict, text_columns: Sequence[str], max_chars: int | None = None) -> str:
    """Join one or more report text columns into a single model prompt input."""
    parts = []
    for column in text_columns:
        value = str(row.get(column, "")).strip()
        if value:
            parts.append(value)
    text = "\n\n".join(parts)
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def stable_text_hash(text: str, length: int = 12) -> str:
    """Return a stable short hash for report text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def build_record_key(id_value: str, report_text: str, date_value: str | None = None) -> str:
    """Build a stable key that distinguishes repeated reports for one patient."""
    text_hash = stable_text_hash(report_text)
    if date_value:
        return f"{id_value}:{date_value}:{text_hash}"
    return f"{id_value}:{text_hash}"


def extract_image_record_id(filename: str | Path) -> str:
    """Extract the source ID while preserving underscores inside that ID."""
    stem = Path(filename).stem
    if "_" not in stem:
        return stem

    prefix, suffix = stem.rsplit("_", 1)
    if HASH_SUFFIX_RE.match(suffix):
        return prefix
    return stem
