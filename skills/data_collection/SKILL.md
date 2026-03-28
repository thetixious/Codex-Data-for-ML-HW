# Skill: Data Collection

## Goal
Collect data from at least 2 sources, unify schema, and generate deep HTML EDA report.

## Minimum deliverables
- At least 2 raw datasets.
- Unified CSV files with metadata columns: `source`, `collected_at`.
- Merged dataset: `data/merged_dataset.csv`.
- EDA HTML report: `reports/eda_report.html`.

## Core scripts
- `scripts/discover_datasets.py`
- `scripts/download_hf.py`
- `scripts/download_kaggle.py`
- `scripts/download_web.py`
- `scripts/unify_and_process.py`
- `scripts/merge_datasets.py`
- `scripts/generate_eda_report.py`

## Notes
- If `data_collection.sources` is empty, run discovery first and pick top relevant candidates for the chosen domain.
- EDA must be domain-aware (not generic only). Use task context.
- Prefer interactive HTML report over static plots-only output.
