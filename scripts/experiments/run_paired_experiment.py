"""Run the frozen 300-case GATED versus UNGATED paired experiment."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from collections import Counter
from pathlib import Path

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from petct.experiment import (  # noqa: E402
    append_audit_event,
    build_work_items,
    load_completed_item_ids,
    load_evaluation_cases,
    prepare_experiment,
    read_work_items,
)
from petct.provenance import sha256_file  # noqa: E402
from petct.provider_config import BindingOverride  # noqa: E402
from webapp.main import (  # noqa: E402
    GeneratePayload,
    RUNTIME_DIR,
    _case_directory,
    _execute_generation,
    _load_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or resume the frozen three-cancer paired experiment. "
            "Completed work items are skipped automatically."
        )
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "evaluation_dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runtime" / "experiments",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-per-cancer", type=int, default=100)
    parser.add_argument("--pipeline", default=None)
    parser.add_argument(
        "--provider-config",
        type=Path,
        help=(
            "Provider/pipeline JSON. Defaults to PETCT_PROVIDER_CONFIG or "
            "settings/providers.json."
        ),
    )
    parser.add_argument(
        "--local-provider-config",
        type=Path,
        help=(
            "Local provider JSON containing stored keys/URLs. Defaults to "
            "PETCT_LOCAL_PROVIDER_CONFIG or settings/local_providers.json."
        ),
    )
    parser.add_argument("--max-revisions", type=int, default=1)
    parser.add_argument(
        "--gated-review-strength",
        choices=["QUICK", "STANDARD", "STRICT"],
        default="STANDARD",
        help="Review strength for GATED runs; UNGATED runs remain OFF.",
    )
    parser.add_argument(
        "--display-finding-types",
        nargs="+",
        choices=[
            "MALIGNANT",
            "BENIGN",
            "INDETERMINATE",
            "IMPORTANT_NEGATIVE",
            "TREATMENT_CONTEXT",
        ],
        default=["MALIGNANT", "INDETERMINATE"],
    )
    parser.add_argument(
        "--display-detail-fields",
        nargs="+",
        choices=["SUVMAX", "FDG_UPTAKE", "LESION_SIZE", "ANATOMICAL_DETAIL"],
        default=["SUVMAX", "FDG_UPTAKE", "LESION_SIZE", "ANATOMICAL_DETAIL"],
    )
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--image-provider")
    parser.add_argument("--image-model")
    parser.add_argument("--review-provider")
    parser.add_argument("--review-model")
    parser.add_argument("--image-size")
    parser.add_argument("--image-quality")
    parser.add_argument("--image-output-format")
    parser.add_argument("--image-background")
    parser.add_argument("--image-compression", type=int)
    parser.add_argument("--image-input-fidelity")
    parser.add_argument("--image-moderation")
    parser.add_argument(
        "--max-items",
        type=int,
        help="Run at most this many pending items; useful for a pilot or staged run.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and lock the experiment without calling any model API.",
    )
    return parser.parse_args()


def _image_override(args: argparse.Namespace) -> BindingOverride | None:
    options = {
        "size": args.image_size,
        "quality": args.image_quality,
        "output_format": args.image_output_format,
        "background": args.image_background,
        "output_compression": args.image_compression,
        "input_fidelity": args.image_input_fidelity,
        "moderation": args.image_moderation,
    }
    options = {key: value for key, value in options.items() if value is not None}
    if not args.image_provider and not args.image_model and not options:
        return None
    return BindingOverride(
        providerId=args.image_provider,
        model=args.image_model,
        options=options,
    )


def _review_override(args: argparse.Namespace) -> BindingOverride | None:
    if not args.review_provider and not args.review_model:
        return None
    return BindingOverride(
        providerId=args.review_provider,
        model=args.review_model,
    )


def _reference_data_url(path: Path | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    suffix_to_mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = suffix_to_mime.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError("reference image must be PNG, JPEG, or WebP")
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", sha256_file(path)


def _selected_pipeline_snapshot(
    pipeline_id: str | None,
    image_override: BindingOverride | None,
    review_override: BindingOverride | None,
) -> dict[str, object]:
    registry = _load_registry()
    public = registry.public_config()
    selected_id = pipeline_id or public["defaultPipelineId"]
    selected = next(
        (pipeline for pipeline in public["pipelines"] if pipeline["id"] == selected_id),
        None,
    )
    if selected is None:
        raise ValueError(f"unknown pipeline: {selected_id}")
    resolved = registry.resolve_pipeline(
        selected_id,
        image_override=image_override,
        review_override=review_override,
        require_review=True,
        require_questions=True,
    )

    def safe_binding(binding) -> dict[str, object]:
        return {
            "providerId": binding.provider_id,
            "providerLabel": binding.provider_label,
            "apiStyle": binding.api_style,
            "baseUrl": binding.base_url,
            "timeoutSeconds": binding.timeout_seconds,
            "model": binding.model,
            "options": binding.options,
            "apiKeyConfigured": bool(binding.api_key),
        }

    return {
        "id": resolved.id,
        "label": resolved.label,
        "image": safe_binding(resolved.image),
        "review": safe_binding(resolved.review),
        "questions": safe_binding(resolved.questions),
        "maxRevisions": resolved.max_revisions,
    }


def _recover_completed_manifests(
    experiment_id: str,
    audit_path: Path,
    completed: set[str],
) -> set[str]:
    """Recover runs saved before a process stopped prior to its audit append."""
    for manifest_path in RUNTIME_DIR.glob("*/*/*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        experiment = manifest.get("experiment") or {}
        item_id = experiment.get("item_id")
        if experiment.get("id") != experiment_id or not item_id or item_id in completed:
            continue
        append_audit_event(
            audit_path,
            {
                "status": "completed",
                "recovered": True,
                "experimentId": experiment_id,
                "itemId": item_id,
                "runId": manifest.get("run_id"),
                "manifestPath": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            },
        )
        completed.add(item_id)
    return completed


def _error_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        return {
            "errorType": type(exc).__name__,
            "detail": detail if isinstance(detail, (str, dict, list)) else str(detail),
        }
    return {
        "errorType": type(exc).__name__,
        "detail": str(exc)[:1000],
    }


async def run(args: argparse.Namespace) -> int:
    if args.provider_config:
        os.environ["PETCT_PROVIDER_CONFIG"] = str(args.provider_config.resolve())
    if args.local_provider_config:
        os.environ["PETCT_LOCAL_PROVIDER_CONFIG"] = str(
            args.local_provider_config.resolve()
        )
    cases, sources = load_evaluation_cases(
        args.dataset_root,
        expected_per_cancer=args.expected_per_cancer,
    )
    work_items = build_work_items(cases, random_seed=args.seed)
    cases_by_pairing_key = {case.pairing_key: case for case in cases}
    image_override = _image_override(args)
    review_override = _review_override(args)
    reference_data_url, reference_sha256 = _reference_data_url(args.reference_image)
    pipeline_snapshot = _selected_pipeline_snapshot(
        args.pipeline,
        image_override,
        review_override,
    )
    selected_pipeline_id = args.pipeline or pipeline_snapshot["id"]
    settings = {
        "pipelineId": selected_pipeline_id,
        "pipelineSnapshot": pipeline_snapshot,
        "maxRevisions": args.max_revisions,
        "generateQuestions": False,
        "gatedReviewStrength": args.gated_review_strength,
        "ungatedReviewStrength": "OFF",
        "displayFindingTypes": args.display_finding_types,
        "displayDetailFields": args.display_detail_fields,
        "inputColumns": {
            "findings": "检查所见",
            "conclusion": "检查结论",
        },
        "imageOverride": (
            image_override.model_dump(by_alias=True) if image_override else None
        ),
        "reviewOverride": (
            review_override.model_dump(by_alias=True) if review_override else None
        ),
        "referenceImage": (
            {
                "fileName": args.reference_image.name,
                "sha256": reference_sha256,
            }
            if args.reference_image
            else None
        ),
    }
    experiment_dir = args.output_root / args.experiment_id
    lock = prepare_experiment(
        experiment_dir=experiment_dir,
        experiment_id=args.experiment_id,
        random_seed=args.seed,
        cases=cases,
        sources=sources,
        work_items=work_items,
        settings=settings,
        project_root=PROJECT_ROOT,
    )
    print(
        f"Locked experiment {args.experiment_id}: "
        f"{lock['plannedCases']} cases, {lock['plannedRuns']} paired runs."
    )
    print(f"Lock: {experiment_dir / 'experiment_lock.json'}")
    if args.prepare_only:
        print("Preparation completed; no model APIs were called.")
        return 0

    audit_path = experiment_dir / "events.jsonl"
    completed = load_completed_item_ids(audit_path)
    completed = _recover_completed_manifests(args.experiment_id, audit_path, completed)
    pending = [
        item
        for item in read_work_items(experiment_dir / "work_items.csv")
        if item.item_id not in completed
    ]
    if args.max_items is not None:
        if args.max_items < 1:
            raise ValueError("--max-items must be at least 1")
        pending = pending[: args.max_items]

    append_audit_event(
        audit_path,
        {
            "status": "session_started",
            "experimentId": args.experiment_id,
            "completedBeforeSession": len(completed),
            "pendingSelected": len(pending),
        },
    )
    failures = 0
    session_counts: Counter[str] = Counter()
    try:
        for position, item in enumerate(pending, start=1):
            case = cases_by_pairing_key[item.pairing_key]
            if case.report_sha256 != item.report_sha256:
                raise ValueError(f"report hash changed for work item {item.item_id}")
            print(
                f"[{position}/{len(pending)}] {item.cancer_type} "
                f"{item.case_id} {item.strategy}"
            )
            append_audit_event(
                audit_path,
                {
                    "status": "started",
                    "experimentId": args.experiment_id,
                    "itemId": item.item_id,
                    "sequence": item.sequence,
                    "pairingKey": item.pairing_key,
                    "caseId": item.case_id,
                    "cancerType": item.cancer_type,
                    "strategy": item.strategy,
                },
            )
            try:
                response = await _execute_generation(
                    GeneratePayload(
                        case_id=item.case_id,
                        report_text=case.report_text,
                        conclusion_text=case.conclusion_text,
                        findings_text=case.findings_text,
                        display_finding_types=args.display_finding_types,
                        display_detail_fields=args.display_detail_fields,
                        gate_enabled=item.strategy == "GATED",
                        review_strength=(
                            args.gated_review_strength
                            if item.strategy == "GATED"
                            else "OFF"
                        ),
                        generate_questions=False,
                        pipeline_id=selected_pipeline_id,
                        image_override=image_override,
                        review_override=review_override,
                        reference_image_data_url=reference_data_url,
                        max_revisions=args.max_revisions,
                        experiment_id=args.experiment_id,
                        experiment_item_id=item.item_id,
                        cancer_type=item.cancer_type,
                        random_seed=args.seed,
                    )
                )
                run_id = response["run_id"]
                manifest_path = _case_directory(run_id) / "manifest.json"
                append_audit_event(
                    audit_path,
                    {
                        "status": "completed",
                        "experimentId": args.experiment_id,
                        "itemId": item.item_id,
                        "runId": run_id,
                        "manifestPath": manifest_path.relative_to(
                            PROJECT_ROOT
                        ).as_posix(),
                        "durationSeconds": response["duration_seconds"],
                    },
                )
                session_counts["completed"] += 1
            except Exception as exc:
                failures += 1
                session_counts["failed"] += 1
                append_audit_event(
                    audit_path,
                    {
                        "status": "failed",
                        "experimentId": args.experiment_id,
                        "itemId": item.item_id,
                        **_error_payload(exc),
                    },
                )
                print(f"  FAILED: {type(exc).__name__}", file=sys.stderr)
    except KeyboardInterrupt:
        append_audit_event(
            audit_path,
            {
                "status": "session_interrupted",
                "experimentId": args.experiment_id,
                **session_counts,
            },
        )
        raise

    append_audit_event(
        audit_path,
        {
            "status": "session_finished",
            "experimentId": args.experiment_id,
            **session_counts,
        },
    )
    total_completed = len(load_completed_item_ids(audit_path))
    print(
        f"Session complete: {session_counts['completed']} completed, "
        f"{failures} failed; experiment total {total_completed}/{len(work_items)}."
    )
    return 1 if failures else 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
