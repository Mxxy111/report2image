# Display Plan and Review Strength Design

## Status

Approved for implementation by user on 2026-06-28.

## Context

The current generation flow accepts a single `report_text`, builds a laterality plan from that text, renders an image prompt, optionally runs AI visual review, and stores the result in `manifest.json`. This has two practical problems for PET/CT reports:

- The "检查结论" section is required for clinical intent, but it can contain items that should not always be drawn.
- Important display details such as SUVmax, lesion size, FDG uptake, and precise anatomy often live in "检查所见", not in "检查结论".

The system needs a controlled AI preprocessing stage that selects what belongs in the patient-facing image before laterality planning and image generation. The system also needs an adjustable AI review strength so outpatient use can trade speed for strictness without changing code.

## Goals

- Split web input into required `检查结论` and optional `检查所见`.
- Add explicit user-selectable content controls for what medical findings and details may appear in the image.
- Add an AI-generated `display_plan` stage before laterality planning.
- Make laterality planning, image generation, and visual review use `display_plan` as the primary contract.
- Add adjustable AI review strength: `OFF`, `QUICK`, `STANDARD`, and `STRICT`.
- Preserve backward compatibility for existing callers that send only `report_text`.
- Record all new settings and generated plans in `manifest.json` and experiment locks.
- Add tests that prove the new contracts, validation, review behavior, and experiment locking.

## Non-Goals

- Do not create a separate 600-case blinded evaluation workbench in this change.
- Do not replace the doctor `PASS/FAIL` gold standard.
- Do not use SUVmax to infer malignancy or change color class.
- Do not allow the model to draw arbitrary findings from "检查所见" unless they match selected finding categories and pass validation.

## User-Facing Design

The web form will replace the single report textarea with:

- `检查结论`: required. This is the primary clinical intent source, but it is still filtered by the display planner.
- `检查所见`: optional. This is used as supporting evidence for SUVmax, lesion size, FDG uptake, and precise anatomy.
- `图中展示的发现类别`: multi-select checkboxes.
- `图中展示的细节字段`: multi-select checkboxes.
- `AI 审查强度`: segmented control with `关闭`, `快速`, `标准`, `严格`.
- `最大修订次数`: existing numeric control, disabled when review strength is `OFF`.

Default finding categories:

- `MALIGNANT`: malignant, suspected malignant, metastasis, recurrence, highly suspicious findings.
- `INDETERMINATE`: uncertain, suspicious, follow-up or clinical-correlation findings.

Default detail fields:

- `SUVMAX`
- `FDG_UPTAKE`
- `LESION_SIZE`
- `ANATOMICAL_DETAIL`

Optional finding categories available but off by default:

- `BENIGN`: benign, inflammatory, cystic, postoperative, low-risk findings.
- `IMPORTANT_NEGATIVE`: clinically important negative statements, such as no recurrence or no distant metastasis.
- `TREATMENT_CONTEXT`: treatment, surgery, or post-therapy context that helps interpret the picture.

At least one finding category must be selected. Detail fields cannot be selected as standalone content because they only modify selected findings.

## API Contract

`GeneratePayload` will add optional fields while preserving existing fields:

```python
conclusion_text: str | None = None
findings_text: str | None = None
display_finding_types: list[DisplayFindingType] | None = None
display_detail_fields: list[DisplayDetailField] | None = None
review_strength: ReviewStrength | None = None
```

Compatibility rules:

- If `conclusion_text` is present, the new display-plan flow is used.
- If `conclusion_text` is absent, the existing `report_text` flow remains available.
- New web requests must send `conclusion_text`; `findings_text` may be empty.
- If `display_finding_types` is missing, backend defaults are applied.
- If `display_detail_fields` is missing, backend defaults are applied.
- If `review_strength` is missing, `gate_enabled=false` maps to `OFF`, and `gate_enabled=true` maps to `STANDARD`.
- `review_strength=OFF` does not require a review provider API key.
- `QUICK`, `STANDARD`, and `STRICT` require a review provider API key.
- Display planning uses the questions/text binding. New display-plan flow requires that binding to be configured.
- In the new flow, `report_text` is derived by joining conclusion and findings with section headers. This derived text is used for compatibility, pairing keys, question generation, and provenance hashes.

## Domain Model

Add `petct/display_plan.py` with enums and Pydantic/domain models.

Finding categories:

```python
DisplayFindingType = Literal[
    "MALIGNANT",
    "BENIGN",
    "INDETERMINATE",
    "IMPORTANT_NEGATIVE",
    "TREATMENT_CONTEXT",
]
```

Detail fields:

```python
DisplayDetailField = Literal[
    "SUVMAX",
    "FDG_UPTAKE",
    "LESION_SIZE",
    "ANATOMICAL_DETAIL",
]
```

Review strength:

```python
ReviewStrength = Literal["OFF", "QUICK", "STANDARD", "STRICT"]
```

`DisplayPlan` contains:

- selected finding categories and detail fields.
- `items`: displayable items, each with stable id, label text, anatomy, patient side, color class, optional SUVmax, optional size, optional FDG uptake, evidence strings, and confidence.
- `excludedItems`: source text that was intentionally excluded and the reason.
- `warnings`: validation or matching concerns that do not necessarily block generation.

Each item must include evidence. SUVmax and lesion size must be traceable to exact source text when present.

## Display Planner

Add an AI display planner provider using the questions/text binding. The planner receives:

- `conclusion_text`
- `findings_text`
- selected finding categories.
- selected detail fields.
- explicit output schema.

The planner returns strict JSON that validates into `DisplayPlan`.

Planner rules:

- Only selected finding categories may become display items.
- Detail fields only attach to already selected findings.
- "检查结论" is required but does not force every conclusion sentence into the image.
- "检查所见" is supporting evidence and cannot freely add unrelated findings.
- SUVmax must appear exactly in source text and must match the relevant finding.
- If multiple SUVmax values could match the same finding, split items if clear; otherwise omit SUVmax and add a warning.
- SUVmax and FDG uptake cannot upgrade benign or uncertain findings to malignant.
- If conclusion and findings conflict, preserve uncertainty rather than resolving it aggressively.

Backend validation after AI output:

- Reject empty display plans with `NO_DISPLAY_ITEMS`; the user should broaden selected finding categories or correct the input text before generating an image.
- Reject SUVmax values not present verbatim in conclusion or findings.
- Reject lesion sizes not present verbatim in conclusion or findings when the size field is set.
- Reject items that have no evidence text.
- Preserve warnings in the manifest.

Generation stops on invalid display-plan output. It does not silently fall back to the old `report_text` behavior.

## Pipeline Changes

New flow:

```text
conclusion_text + findings_text + selected content
        -> AI display_plan
        -> laterality_plan from display_plan items
        -> image prompt using display_plan as primary contract
        -> AI review according to review_strength
        -> optional image edit revisions
        -> manifest
```

Legacy flow:

```text
report_text
        -> current laterality_plan
        -> current image prompt
        -> current AI review when gate_enabled=true
        -> manifest
```

The new `GenerationRequest` will carry:

- `report_text`: compatibility rendering of conclusion and findings.
- `conclusion_text`
- `findings_text`
- `display_plan`
- `laterality_plan`
- `review_strength`

Laterality planning in the new flow uses display-plan items instead of full raw report text. This keeps the laterality planner focused on findings that will actually appear in the image.

Patient-comprehension questions in the new flow use the derived report text plus display-plan summary. This keeps question generation aligned with what appears in the patient-facing image while retaining enough report context for clinically sensible distractors.

## Image Prompt Changes

The image prompt will state:

- Draw only `display_plan.items` where `draw=true`.
- Use `labelText` as the primary visible label.
- Do not add findings, diagnoses, SUVmax values, dimensions, organs, or abnormalities outside the display plan.
- Use original conclusion and findings only as supporting evidence.
- Keep existing patient-friendly visual style, color semantics, laterality mapping, and SUVmax constraints.

## AI Review Strength

`OFF`:

- No AI review call.
- No review provider required.
- `gate_outcome.status = NOT_REVIEWED`.

`QUICK`:

- One comprehensive visual review call per attempt.
- Skips independent laterality precheck.
- Only blocks major errors: obvious laterality reversal, key omission, dangerous hallucination, major anatomy error, wrong malignancy/color, critical SUVmax mismatch, or unreadable key text.

`STANDARD`:

- Current default behavior.
- Runs laterality precheck when display/laterality plan contains sided endpoint items.
- Runs comprehensive review after laterality passes or when precheck is not applicable.
- Blocks major errors while allowing patient-friendly grouping of complex, bilateral, or multifocal findings.

`STRICT`:

- Runs standard laterality precheck.
- Uses a stricter comprehensive prompt.
- Also blocks ambiguous key labels, unclear SUVmax matching, weakened uncertainty qualifiers, confusing display-plan omissions, and image additions outside the display plan.

Review outputs remain structured as `passed` and `reason`. Future typed error categories can be added later without changing this feature.

## Manifest and Experiment Lock

`manifest.json` will include:

```json
{
  "input_texts": {
    "conclusion_text": "...",
    "findings_text": "..."
  },
  "display_finding_types": ["MALIGNANT", "INDETERMINATE"],
  "display_detail_fields": ["SUVMAX", "FDG_UPTAKE", "LESION_SIZE", "ANATOMICAL_DETAIL"],
  "display_plan": {},
  "display_plan_prompt_version": "petct-display-plan-2026-06-28.1",
  "review_strength": "STANDARD"
}
```

`experiment_lock.json` will freeze:

- selected finding categories.
- selected detail fields.
- review strength.
- display-plan prompt version and hash.
- input source columns for conclusion and findings.

Exported statistical data will not include full conclusion, findings, or evidence text. It will include frozen configuration values and hashes needed for audit.

## Error Handling

Display-planning errors use structured HTTP details:

- missing conclusion text.
- no finding category selected.
- missing text model configuration.
- upstream display planner failure.
- invalid planner JSON.
- unverifiable SUVmax or lesion size.
- empty display plan for selected categories.

Review-strength errors:

- `QUICK`, `STANDARD`, or `STRICT` without review provider configuration returns a configuration error.
- `OFF` ignores max revisions and disables review UI controls.

## Security and Safety

- Raw report text and AI planner output are untrusted data.
- Prompt wording must treat report content as source data, not instructions.
- AI planner output is validated before it can influence prompts, laterality, or review.
- API keys remain in environment or local provider files and are never exposed through public config responses.
- Manifest may contain sensitive report-derived text and remains under `runtime/`.

## Testing Plan

Unit tests:

- payload compatibility: old `report_text` still works.
- new payload requires `conclusion_text`.
- finding/detail selection validation rejects detail-only selections.
- display-plan validation rejects SUVmax not present in source text.
- display-plan validation preserves excluded items and warnings.
- review strength maps from legacy `gate_enabled`.
- `QUICK` skips laterality precheck.
- `STANDARD` preserves current two-step review behavior.
- `OFF` avoids review provider requirement and returns not reviewed.

Integration tests:

- `_execute_generation` records input texts, display selections, display plan, and review strength in manifest.
- questions still run from the appropriate report context.
- batch experiment lock changes when display selections or review strength change.
- export includes new frozen config columns without exporting full report text.

Frontend checks:

- page contains conclusion and findings inputs.
- content category and detail checkboxes exist with expected defaults.
- review strength segmented control exists and disables review controls when `OFF`.
- generated payload contains new fields.

Verification commands:

```powershell
node --check webapp/static/app.js
python -B -m pytest --basetemp .tmp\pytest-<timestamp> -p no:cacheprovider -q
```

## Rollout

Implement in small vertical slices:

1. Display-plan models and validation.
2. Display planner provider and prompt.
3. Backend payload and execution flow.
4. Review-strength behavior.
5. Manifest, experiment lock, and export changes.
6. Web UI changes.
7. Documentation updates and full verification.

## Acceptance Criteria

- New web users can enter `检查结论` and optional `检查所见`.
- Users can choose display finding categories and detail fields.
- Users can choose AI review strength.
- New-flow generation produces a validated `display_plan`.
- Laterality, image prompt, and AI review use the display plan.
- SUVmax is shown only when selected and verifiable in source text.
- Legacy `report_text` requests remain supported.
- Manifest and experiment locks capture all new settings.
- Tests and JavaScript syntax check pass.
