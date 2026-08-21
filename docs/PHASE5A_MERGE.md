# Phase 5A — Feedback Ingest & Dataset Merge

> Close the online→offline loop: promote executed-query feedback into the
> training set, safely and idempotently, behind a hard validation gate.

## What this subphase delivers

| Artifact | Purpose |
|---|---|
| `services/automl_service/merge.py` | Merge `data/feedback/fb_*.json` into `data/processed/features.csv`, deduped |
| `services/automl_service/watermark.py` | Persistent bookmark so the scheduler knows when new data exists |
| `scripts/merge_feedback.py` | CLI: dry-run / `--apply` / `--gate` |
| `tests/test_feedback_merge.py` | Unit tests for the dedup core + watermark |

## The pipeline

```
data/feedback/fb_*.json
        │  extract_feedback()  (reuses the offline feature pipeline — no drift)
        ▼
   feedback rows (DataFrame)
        │  dedupe_new_rows(base, feedback)      ← pure, unit-tested
        ▼
   NEW rows only
        │  --apply
        ▼
data/processed/features.csv  (rewritten)  →  merge_watermark.json bumped
        │  --gate
        ▼
scripts/validate_dataset.py --features   (HARD GATE, exit≠0 ⇒ abort retrain)
```

## Design decisions — what / why / why-not

**Reuse `feedback_to_features.extract_feedback` instead of re-implementing feature math.**
Why: the single most damaging class of bug in a learned optimizer is
online/offline *feature skew* — the model trains on one representation and
serves on another. Reusing the exact offline extractor makes skew
structurally impossible. Why-not a fresh extractor: it would drift the day
someone edits one copy and forgets the other.

**Idempotent dedup keyed on `(sql_hash, variant, collected_at)`.**
Why: the worker (5E) may re-run a merge after a crash, and feedback files
are append-only. Keying on identity means "merge twice = merge once". Why
those three columns: `sql_hash+variant` identifies the plan; `collected_at`
distinguishes repeated executions of the same plan over time (which are
legitimately new training rows).

**Pure `dedupe_new_rows(base, feedback)` separated from disk I/O.**
Why: it lets us unit-test the only tricky logic with tiny DataFrames — no
Postgres, no plan JSON, no feature extraction — so CI stays fast and the
tests actually run on every push.

**Validation gate is a subprocess, not an import.**
Why: `validate_dataset.py` is a *gate with an exit code*. Shelling out gives
clean process isolation and a single source of truth for "is this dataset
trustworthy?", identical to how a human runs it. A merge that would inject
corrupt or schema-mismatched rows (e.g. the TPC-H/TPC-DS `customer`
collision) aborts the retrain instead of silently poisoning the model.

**Watermark is advisory, not authoritative.**
Why: losing or corrupting it must never cause data loss. Because the merge
itself is idempotent, a missing watermark just means "re-scan everything and
dedupe". The watermark exists only to answer the scheduler's cheap question
*"is there enough new feedback to bother retraining?"* (see 5D).

## Usage

```bash
# See what would be merged, change nothing:
python scripts/merge_feedback.py

# Merge for real:
python scripts/merge_feedback.py --apply

# Merge, then refuse to proceed if the dataset fails validation:
python scripts/merge_feedback.py --apply --gate
```

`MergeReport` (printed as JSON) fields: `scanned_feedback_files`,
`feedback_rows`, `new_rows`, `duplicates`, `features_before`,
`features_after`, `applied`.

## Acceptance criteria

- Running the merge twice adds rows once (idempotent). ✔ tested
- Dedup keeps genuinely-new `(hash, variant, time)` rows, drops repeats. ✔ tested
- Watermark round-trips, accumulates, and self-heals on corruption. ✔ tested
- `--gate` returns exit code `2` when validation fails (retrain aborts).

## What is intentionally deferred

- Streaming merge straight from Kafka (today we merge the consumer's
  on-disk output — same records, simpler failure model).
- Feature-store integration; `features.csv` remains the source of truth for
  Phase 5.
