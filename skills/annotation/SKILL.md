# Skill: Annotation + HITL

## Goal
Auto-label dataset with confidence, stop for mandatory manual review, and merge reviewed labels.

## Minimum deliverables
- `data/auto_labeled_dataset.csv`
- `data/review_queue.csv` (confidence < threshold)
- `data/labeled_dataset.csv` (after merge)
- `labeling/annotation_spec.md`
- `labeling/labelstudio_import.json`
- `reports/annotation_metrics.json`

## Core scripts
- `scripts/auto_label.py`
- `scripts/build_review_queue.py`
- `scripts/merge_reviewed.py`
- `scripts/generate_spec.py`
- `scripts/check_quality.py`
- `scripts/export_to_labelstudio.py`

## Required interaction
- Stop after building `review_queue.csv`.
- Wait for user confirmation that manual corrections are finished.
- Continue only after confirmation.
