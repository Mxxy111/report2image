"""Contracts for a resumable, auditable paired PET-CT experiment."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from petct.data import (
    build_record_key,
    compose_report_text,
    detect_id_column,
    detect_text_columns,
    read_table,
)
from petct.provenance import (
    git_provenance,
    sha256_file,
    sha256_text,
    source_tree_sha256,
)
from petct.providers import (
    DISPLAY_PLAN_PROMPT_VERSION,
    OpenAIDisplayPlanner,
    OpenAIVisualReviewer,
    REVIEW_PROMPT_VERSION,
)
from petct.question_service import OpenAIQuestionService, QUESTION_PROMPT_VERSION
from prompts import (
    IMAGE_PROMPT_VERSION,
    PET_CT_IMG2IMG_PROMPT,
    PET_CT_VISUALIZATION_PROMPT,
)


CANCER_TYPES = ("肾癌", "前列腺癌", "尿路上皮癌")
STRATEGIES = ("UNGATED", "GATED")


@dataclass(frozen=True)
class DatasetSource:
    cancer_type: str
    path: str
    sha256: str
    row_count: int


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    cancer_type: str
    report_text: str
    findings_text: str
    conclusion_text: str
    report_date: str
    pairing_key: str
    report_sha256: str
    source_file: str
    source_row: int


@dataclass(frozen=True)
class WorkItem:
    sequence: int
    item_id: str
    pairing_key: str
    case_id: str
    cancer_type: str
    strategy: Literal["UNGATED", "GATED"]
    report_sha256: str
    source_file: str
    source_row: int


def load_evaluation_cases(
    dataset_root: str | Path,
    *,
    expected_per_cancer: int = 100,
) -> tuple[list[EvaluationCase], list[DatasetSource]]:
    """Load and validate the frozen three-cancer evaluation cohort."""
    root = Path(dataset_root)
    cases: list[EvaluationCase] = []
    sources: list[DatasetSource] = []

    for cancer_type in CANCER_TYPES:
        candidates = sorted((root / cancer_type).glob("*.csv"))
        if len(candidates) != 1:
            raise ValueError(
                f"{cancer_type} requires exactly one cohort CSV; found {len(candidates)}"
            )
        path = candidates[0]
        table = read_table(path)
        if len(table) != expected_per_cancer:
            raise ValueError(
                f"{cancer_type} requires {expected_per_cancer} cases; found {len(table)}"
            )
        id_column = detect_id_column(list(table.columns))
        text_columns = detect_text_columns(
            list(table.columns),
            requested=["检查所见", "检查结论"],
        )
        date_column = "PETCT检查日期" if "PETCT检查日期" in table.columns else None
        relative_path = path.relative_to(root).as_posix()
        sources.append(
            DatasetSource(
                cancer_type=cancer_type,
                path=relative_path,
                sha256=sha256_file(path),
                row_count=len(table),
            )
        )

        for zero_based_row, row in table.iterrows():
            row_data = row.to_dict()
            case_id = str(row_data.get(id_column, "")).strip()
            findings_text = str(row_data.get("检查所见", "")).strip()
            conclusion_text = str(row_data.get("检查结论", "")).strip()
            report_text = compose_report_text(row_data, text_columns)
            report_date = (
                str(row_data.get(date_column, "")).strip() if date_column else ""
            )
            if not case_id or not report_text or not conclusion_text:
                raise ValueError(
                    f"{relative_path} row {zero_based_row + 2} lacks case ID, report text, or conclusion text"
                )
            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    cancer_type=cancer_type,
                    report_text=report_text,
                    findings_text=findings_text,
                    conclusion_text=conclusion_text,
                    report_date=report_date,
                    pairing_key=build_record_key(case_id, report_text, report_date or None),
                    report_sha256=sha256_text(report_text),
                    source_file=relative_path,
                    source_row=zero_based_row + 2,
                )
            )

    pairing_keys = [case.pairing_key for case in cases]
    if len(set(pairing_keys)) != len(pairing_keys):
        duplicates = [
            key for key, count in Counter(pairing_keys).items() if count > 1
        ]
        raise ValueError(f"cohort contains duplicate pairing keys: {duplicates[:5]}")
    return cases, sources


def build_work_items(
    cases: list[EvaluationCase],
    *,
    random_seed: int,
) -> list[WorkItem]:
    """Randomize case and within-pair strategy order reproducibly."""
    rng = random.Random(random_seed)
    ordered_cases = list(cases)
    rng.shuffle(ordered_cases)
    planned: list[tuple[EvaluationCase, str]] = []
    for case in ordered_cases:
        strategies = list(STRATEGIES)
        rng.shuffle(strategies)
        planned.extend((case, strategy) for strategy in strategies)

    work_items = []
    for sequence, (case, strategy) in enumerate(planned, start=1):
        item_material = f"{random_seed}\0{case.pairing_key}\0{strategy}"
        work_items.append(
            WorkItem(
                sequence=sequence,
                item_id=sha256_text(item_material)[:20],
                pairing_key=case.pairing_key,
                case_id=case.case_id,
                cancer_type=case.cancer_type,
                strategy=strategy,
                report_sha256=case.report_sha256,
                source_file=case.source_file,
                source_row=case.source_row,
            )
        )
    return work_items


def _prompt_lock() -> dict[str, object]:
    return {
        "image": {
            "version": IMAGE_PROMPT_VERSION,
            "textToImageTemplateSha256": sha256_text(PET_CT_VISUALIZATION_PROMPT),
            "referenceImageTemplateSha256": sha256_text(PET_CT_IMG2IMG_PROMPT),
        },
        "displayPlan": {
            "version": DISPLAY_PLAN_PROMPT_VERSION,
            "sha256": sha256_text(OpenAIDisplayPlanner.INSTRUCTIONS),
        },
        "review": {
            "version": REVIEW_PROMPT_VERSION,
            "sha256": sha256_text(OpenAIVisualReviewer.REVIEW_INSTRUCTIONS),
        },
        "questions": {
            "version": QUESTION_PROMPT_VERSION,
            "sha256": sha256_text(OpenAIQuestionService.INSTRUCTIONS),
        },
    }


def _cohort_lock(cases: list[EvaluationCase]) -> list[dict[str, object]]:
    return [
        {
            "caseId": case.case_id,
            "cancerType": case.cancer_type,
            "pairingKey": case.pairing_key,
            "reportDate": case.report_date,
            "reportSha256": case.report_sha256,
            "findingsSha256": sha256_text(case.findings_text),
            "conclusionSha256": sha256_text(case.conclusion_text),
            "sourceFile": case.source_file,
            "sourceRow": case.source_row,
        }
        for case in sorted(cases, key=lambda item: item.pairing_key)
    ]


def _protocol_lock(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root)
    documents = {
        "statisticalAnalysisPlan": root / "docs" / "statistical_analysis_plan.md",
        "imageEvaluationPlan": root / "docs" / "image_evaluation_plan.md",
    }
    return {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in documents.items()
        if path.is_file()
    }


def prepare_experiment(
    *,
    experiment_dir: str | Path,
    experiment_id: str,
    random_seed: int,
    cases: list[EvaluationCase],
    sources: list[DatasetSource],
    work_items: list[WorkItem],
    settings: dict[str, object],
    project_root: str | Path,
) -> dict[str, object]:
    """Create an immutable experiment lock, or validate it for resume."""
    directory = Path(experiment_dir)
    lock_path = directory / "experiment_lock.json"
    work_items_path = directory / "work_items.csv"
    cancer_counts = Counter(case.cancer_type for case in cases)
    comparable = {
        "schemaVersion": 1,
        "experimentId": experiment_id,
        "design": "paired retrospective GATED versus UNGATED",
        "randomSeed": random_seed,
        "randomSeedPurpose": "case order and within-pair strategy order",
        "providerSeedApplied": False,
        "plannedCases": len(cases),
        "plannedRuns": len(work_items),
        "cancerCounts": dict(sorted(cancer_counts.items())),
        "datasetSources": [asdict(source) for source in sources],
        "cohort": _cohort_lock(cases),
        "settings": settings,
        "prompts": _prompt_lock(),
        "protocol": _protocol_lock(project_root),
        "software": {
            "sourceTreeSha256": source_tree_sha256(project_root),
            "git": git_provenance(project_root),
        },
    }

    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        existing_comparable = {
            key: value for key, value in existing.items() if key != "createdAt"
        }
        if existing_comparable != comparable:
            raise ValueError(
                "locked experiment differs from current cohort, settings, prompts, "
                "seed, or software; create a new experiment ID"
            )
        if not work_items_path.exists():
            raise ValueError("locked experiment is missing work_items.csv")
        return existing

    directory.mkdir(parents=True, exist_ok=True)
    lock = {
        **comparable,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with work_items_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(asdict(work_items[0]).keys()) if work_items else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(asdict(item) for item in work_items)
    append_audit_event(
        directory / "events.jsonl",
        {
            "status": "experiment_locked",
            "experimentId": experiment_id,
            "plannedCases": len(cases),
            "plannedRuns": len(work_items),
        },
    )
    return lock


def read_work_items(path: str | Path) -> list[WorkItem]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            WorkItem(
                sequence=int(row["sequence"]),
                item_id=row["item_id"],
                pairing_key=row["pairing_key"],
                case_id=row["case_id"],
                cancer_type=row["cancer_type"],
                strategy=row["strategy"],
                report_sha256=row["report_sha256"],
                source_file=row["source_file"],
                source_row=int(row["source_row"]),
            )
            for row in rows
        ]


def append_audit_event(path: str | Path, event: dict[str, object]) -> None:
    """Append one immutable JSON line and flush it to disk."""
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with audit_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def load_completed_item_ids(path: str | Path) -> set[str]:
    audit_path = Path(path)
    if not audit_path.exists():
        return set()
    completed: set[str] = set()
    with audit_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid audit JSON at {audit_path}:{line_number}"
                ) from exc
            if event.get("status") == "completed" and event.get("itemId"):
                completed.add(str(event["itemId"]))
    return completed


def _completed_audit_events(path: Path) -> dict[str, dict[str, object]]:
    events: dict[str, dict[str, object]] = {}
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid audit JSON at {path}:{line_number}") from exc
            if event.get("status") != "completed" or not event.get("itemId"):
                continue
            item_id = str(event["itemId"])
            previous = events.get(item_id)
            if (
                previous
                and previous.get("runId")
                and event.get("runId")
                and previous["runId"] != event["runId"]
            ):
                raise ValueError(f"work item {item_id} has multiple completed run IDs")
            events[item_id] = event
    return events


def export_experiment_results(
    *,
    experiment_dir: str | Path,
    project_root: str | Path,
    output_path: str | Path,
) -> list[dict[str, object]]:
    """Export one audit row per planned work item without report text or names."""
    directory = Path(experiment_dir)
    root = Path(project_root)
    lock = json.loads(
        (directory / "experiment_lock.json").read_text(encoding="utf-8")
    )
    events = _completed_audit_events(directory / "events.jsonl")
    rows: list[dict[str, object]] = []

    for item in read_work_items(directory / "work_items.csv"):
        event = events.get(item.item_id)
        base = {
            "experiment_id": lock["experimentId"],
            "sequence": item.sequence,
            "item_id": item.item_id,
            "pairing_key": item.pairing_key,
            "case_id": item.case_id,
            "cancer_type": item.cancer_type,
            "strategy": item.strategy,
            "run_status": "MISSING",
            "run_id": "",
            "manifest_path": "",
            "generated_at": "",
            "duration_seconds": "",
            "gate_accepted": "",
            "gate_status": "",
            "attempt_count": "",
            "revision_count": "",
            "image_provider": "",
            "image_api_style": "",
            "image_base_url": "",
            "image_timeout_seconds": "",
            "image_model": "",
            "image_options_json": "",
            "review_provider": "",
            "review_api_style": "",
            "review_base_url": "",
            "review_timeout_seconds": "",
            "review_model": "",
            "max_revisions": "",
            "review_strength": "",
            "display_finding_types": "",
            "display_detail_fields": "",
            "display_item_count": "",
            "display_plan_sha256": "",
            "human_decision": "",
            "human_error_types": "",
            "reviewer": "",
            "notes": "",
            "image_prompt_version": "",
            "image_prompt_sha256": "",
            "display_plan_prompt_version": "",
            "display_plan_prompt_sha256": "",
            "review_prompt_version": "",
            "review_prompt_sha256": "",
            "reference_image_sha256": "",
            "source_findings_column": (
                ((lock.get("settings") or {}).get("inputColumns") or {}).get(
                    "findings", ""
                )
            ),
            "source_conclusion_column": (
                ((lock.get("settings") or {}).get("inputColumns") or {}).get(
                    "conclusion", ""
                )
            ),
            "source_tree_sha256": "",
            "random_seed": lock.get("randomSeed", ""),
            "provider_seed_applied": lock.get("providerSeedApplied", False),
        }
        if event:
            manifest_relative = str(event.get("manifestPath", ""))
            manifest_path = root / manifest_relative
            if not manifest_path.is_file():
                raise ValueError(
                    f"completed item {item.item_id} is missing manifest {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("pairing_key") != item.pairing_key:
                raise ValueError(f"pairing key mismatch for work item {item.item_id}")
            if manifest.get("strategy") != item.strategy:
                raise ValueError(f"strategy mismatch for work item {item.item_id}")
            evaluation = manifest.get("human_evaluation") or {}
            reproducibility = manifest.get("reproducibility") or {}
            prompts = reproducibility.get("prompts") or {}
            image_prompt = prompts.get("image") or {}
            display_prompt = prompts.get("displayPlan") or {}
            review_prompt = prompts.get("review") or {}
            pipeline = manifest.get("pipeline") or {}
            image_binding = pipeline.get("image") or {}
            review_binding = pipeline.get("review") or {}
            gate_outcome = manifest.get("gate_outcome") or {}
            display_selection = manifest.get("display_selection") or {}
            display_plan = manifest.get("display_plan")
            display_plan_sha256 = (
                sha256_text(
                    json.dumps(display_plan, ensure_ascii=False, sort_keys=True)
                )
                if display_plan
                else ""
            )
            base.update(
                {
                    "run_status": "COMPLETED",
                    "run_id": manifest.get("run_id", ""),
                    "manifest_path": manifest_relative,
                    "generated_at": manifest.get("generated_at", ""),
                    "duration_seconds": manifest.get("duration_seconds", ""),
                    "gate_accepted": manifest.get("accepted", ""),
                    "gate_status": gate_outcome.get("status", ""),
                    "attempt_count": gate_outcome.get("attemptCount", ""),
                    "revision_count": gate_outcome.get("revisionCount", ""),
                    "image_provider": image_binding.get("provider", ""),
                    "image_api_style": image_binding.get("api_style", ""),
                    "image_base_url": image_binding.get("base_url", ""),
                    "image_timeout_seconds": image_binding.get(
                        "timeout_seconds", ""
                    ),
                    "image_model": image_binding.get("model", ""),
                    "image_options_json": json.dumps(
                        image_binding.get("options", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "review_provider": review_binding.get("provider", ""),
                    "review_api_style": review_binding.get("api_style", ""),
                    "review_base_url": review_binding.get("base_url", ""),
                    "review_timeout_seconds": review_binding.get(
                        "timeout_seconds", ""
                    ),
                    "review_model": review_binding.get("model", ""),
                    "max_revisions": pipeline.get("max_revisions", ""),
                    "review_strength": manifest.get("review_strength", ""),
                    "display_finding_types": "|".join(
                        display_selection.get("finding_types", [])
                    ),
                    "display_detail_fields": "|".join(
                        display_selection.get("detail_fields", [])
                    ),
                    "display_item_count": len((display_plan or {}).get("items", [])),
                    "display_plan_sha256": display_plan_sha256,
                    "human_decision": evaluation.get("overallDecision", ""),
                    "human_error_types": "|".join(evaluation.get("errorTypes", [])),
                    "reviewer": evaluation.get("reviewer", ""),
                    "notes": evaluation.get("notes", ""),
                    "image_prompt_version": image_prompt.get("version", ""),
                    "image_prompt_sha256": image_prompt.get("sha256", ""),
                    "display_plan_prompt_version": display_prompt.get("version", ""),
                    "display_plan_prompt_sha256": display_prompt.get("sha256", ""),
                    "review_prompt_version": review_prompt.get("version", ""),
                    "review_prompt_sha256": review_prompt.get("sha256", ""),
                    "reference_image_sha256": reproducibility.get(
                        "referenceImageSha256", ""
                    )
                    or "",
                    "source_tree_sha256": (
                        reproducibility.get("software") or {}
                    ).get("sourceTreeSha256", ""),
                    "random_seed": reproducibility.get(
                        "randomSeed", lock.get("randomSeed", "")
                    ),
                    "provider_seed_applied": reproducibility.get(
                        "providerSeedApplied",
                        lock.get("providerSeedApplied", False),
                    ),
                }
            )
        rows.append(base)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows
