"""Publication-oriented provenance for generated PET-CT artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from petct.providers import (
    DISPLAY_PLAN_PROMPT_VERSION,
    LATERALITY_PLANNER_PROMPT_VERSION,
    OpenAIDisplayPlanner,
    OpenAILateralityPlanner,
    OpenAIVisualReviewer,
    REVIEW_PROMPT_VERSION,
)
from petct.question_service import OpenAIQuestionService, QUESTION_PROMPT_VERSION
from prompts import IMAGE_PROMPT_VERSION


SOURCE_DIRS = ("petct", "webapp", "scripts")
SOURCE_FILES = (
    "cli.py",
    "client.py",
    "config.py",
    "processor.py",
    "prompts.py",
    "run_web.py",
    "pyproject.toml",
    "requirements.txt",
    "settings/providers.json",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_text(content: str) -> str:
    return sha256_bytes(content.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(project_root: str | Path) -> str:
    """Hash publication-relevant source/config files in stable path order."""
    root = Path(project_root)
    paths: set[Path] = set()
    for directory in SOURCE_DIRS:
        source_dir = root / directory
        if source_dir.exists():
            paths.update(source_dir.rglob("*.py"))
    paths.update(root / name for name in SOURCE_FILES if (root / name).is_file())

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_provenance(project_root: str | Path) -> dict[str, object]:
    """Return commit state without failing when this project is not tracked."""
    root = Path(project_root)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )

    tracked_check = run("ls-files", "--error-unmatch", "--", "pyproject.toml")
    if tracked_check.returncode != 0:
        return {"tracked": False, "commit": None, "dirty": True}

    commit_result = run("rev-parse", "HEAD")
    status_result = run("status", "--porcelain", "--", ".")
    return {
        "tracked": True,
        "commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
    }


def build_reproducibility_metadata(
    *,
    project_root: str | Path,
    image_prompt_template: str,
    rendered_image_prompt: str,
    random_seed: int | None,
    reference_image: bytes | None,
) -> dict[str, object]:
    """Return a secret-free reproducibility snapshot for one generated run."""
    return {
        "schemaVersion": 1,
        "prompts": {
            "displayPlan": {
                "version": DISPLAY_PLAN_PROMPT_VERSION,
                "sha256": sha256_text(OpenAIDisplayPlanner.INSTRUCTIONS),
            },
            "laterality": {
                "version": LATERALITY_PLANNER_PROMPT_VERSION,
                "sha256": sha256_text(OpenAILateralityPlanner.INSTRUCTIONS),
            },
            "image": {
                "version": IMAGE_PROMPT_VERSION,
                "templateSha256": sha256_text(image_prompt_template),
                "sha256": sha256_text(rendered_image_prompt),
            },
            "review": {
                "version": REVIEW_PROMPT_VERSION,
                "sha256": sha256_text(OpenAIVisualReviewer.instruction_bundle()),
            },
            "questions": {
                "version": QUESTION_PROMPT_VERSION,
                "sha256": sha256_text(OpenAIQuestionService.INSTRUCTIONS),
            },
        },
        "randomSeed": random_seed,
        "randomSeedPurpose": (
            "batch work-item ordering and identification"
            if random_seed is not None
            else None
        ),
        # OpenAI-compatible image APIs used here do not expose a seed parameter.
        "providerSeedApplied": False,
        "referenceImageSha256": (
            sha256_bytes(reference_image) if reference_image else None
        ),
        "software": {
            "sourceTreeSha256": source_tree_sha256(project_root),
            "git": git_provenance(project_root),
        },
    }
