# Codex Data Pipeline HW


## Агенты из домашек + фичи
- `agents/data_collection_agent.py`
- `agents/data_quality_agent.py`
- `agents/annotation_agent.py`
- `agents/al_agent.py`
- `run_pipeline.py` (единый запуск одной командой)
- `notebooks/eda.ipynb`
- `notebooks/data_quality.ipynb`
cnjg- `notebooks/al_experiment.ipynb`
- `data/raw`, `data/labeled`, `models`, `reports`, `review_queue.csv`

## Что реализовано
- Автопоиск релевантных датасетов под произвольную тему (`hf/kaggle/zenodo`) с scoring.
- Multi-source data collection (HF/Kaggle/Web).
- Unified schema + merge.
- EDA в виде **интерактивного HTML-отчета**.
- Data quality аудит: отчеты `md + html + json`, очистка и сравнение до/после.
- Annotation + HITL: авторазметка, `review_queue.csv`, merge после ручной правки, LabelStudio export.
- Active Learning: `entropy vs random`, histories, learning curves (`png + html`), savings conclusion.
- Изоляция прогонов: `data/raw/<task>_<timestamp>/...`.

## Структура
- `skills/` — этапы и скрипты.
- `utils/` — общие утилиты.
- `config.yaml` — базовые настройки.
- `data/raw/` — артефакты прогонов.

## Установка
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Доступ к источникам
- HuggingFace (опционально): `export HF_TOKEN=...`
- Kaggle:
  - стандартно: `export KAGGLE_USERNAME=...` и `export KAGGLE_KEY=...`
  - поддерживается fallback: `export KAGGLE_API_TOKEN=...`
    (форматы: JSON `{"username":"...","key":"..."}`, `username:key`, или `key` при заданном `KAGGLE_USERNAME`)
  - также поддерживается локальный файл `.envar` в корне проекта (подхватывается автоматически в discovery/download):
    - `KAGGLE_API_TOKEN=...`
    - `KAGGLE_USERNAME=...` (если токен key-only)

## Быстрый сценарий запуска (end-to-end)
Из корня проекта `Codex-Data-for-ML-HW`:

### Вариант 1: Формальный запуск для проверки
```bash
python run_pipeline.py --topic "mortgage risk prediction" --auto_approve
```

Результаты:
- `data/raw/collected_unified.csv`, `data/raw/cleaned.csv`
- `data/labeled/labeled_dataset.csv`
- `data/labeled/data_card.md`
- `review_queue.csv` (HITL)
- `models/final_model.pkl`
- `reports/eda_report.html`, `reports/quality_report.html`, `reports/al_report.html`
- `reports/quality_report.md`, `reports/annotation_report.md`, `reports/al_report.md`
- `reports/final_report.md`

В skill-orchestrator run-root дополнительно создаются:
- `reports/al_learning_curves.html` (улучшенный AL-отчет)
- `reports/final_report.html` (финальный отчет в HTML)

### Codex CLI Skill-first режим (рекомендуется)
▌ Активируй `@skills/codex_cli_pipeline/SKILL.md` и проведи меня по пайплайну для темы:  ...


Файл skill:
- [skills/codex_cli_pipeline/SKILL.md](skills/codex_cli_pipeline/SKILL.md)

### Вариант A: Single-entry (одна команда)
Полный запуск через оркестратор:
```bash
python skills/orchestrator/scripts/run_pipeline.py --topic "fraud detection in transactions" --auto_confirm
```

Интерактивно с подтверждениями и HITL-паузой:
```bash
python skills/orchestrator/scripts/run_pipeline.py --topic "toxic comments classification"
```

Демонстрационный non-interactive прогон (без остановки на review):
```bash
python skills/orchestrator/scripts/run_pipeline.py --topic "customer churn prediction" --auto_confirm --no_review_pause
```

Запуск подмножества этапов для существующего прогона:
```bash
python skills/orchestrator/scripts/run_pipeline.py \
  --run_root "<path_to_existing_run>" \
  --from_stage annotation \
  --to_stage final_report
```

### Вариант B: Модульный (по скриптам)

1. Создать run-папку:
```bash
RUN_ROOT=$(python skills/orchestrator/scripts/init_run.py --task_name "my-domain-task")
```

2. Discovery (для любой области):
```bash
python skills/data_collection/scripts/discover_datasets.py \
  --topic "my domain topic" \
  --output_json "$RUN_ROOT/reports/discovery_candidates.json"
```

3. Сбор данных (пример ручного режима):
```bash
python skills/data_collection/scripts/download_hf.py \
  --name "nehalbirla/vehicle-dataset-from-cardekho" \
  --output_dir "$RUN_ROOT/data"

python skills/data_collection/scripts/download_kaggle.py \
  --name "<owner>/<dataset>" \
  --output_dir "$RUN_ROOT/data"
```

4. Унификация и merge (пример с явным mapping):
```bash
python skills/data_collection/scripts/unify_and_process.py \
  --input_path "$RUN_ROOT/data/<raw1>.csv" \
  --output_path "$RUN_ROOT/data/unified_1.csv" \
  --rename_map '{"selling_price":"price","year":"year","km_driven":"mileage","name":"model","fuel":"fuel_type","transmission":"transmission"}' \
  --keep_cols '["model","year","price","mileage","fuel_type","transmission"]' \
  --source_name "source_1" \
  --run_timestamp "$(date +%Y-%m-%d_%H-%M)"

python skills/data_collection/scripts/merge_datasets.py \
  --input_dir "$RUN_ROOT/data" \
  --output_file "$RUN_ROOT/data/merged_dataset.csv" \
  --pattern "unified_*.csv"
```

5. EDA HTML:
```bash
python skills/data_collection/scripts/generate_eda_report.py \
  --input_csv "$RUN_ROOT/data/merged_dataset.csv" \
  --output_html "$RUN_ROOT/reports/eda_report.html" \
  --task_description "car price prediction"
```

6. Data quality:
```bash
python skills/data_quality/scripts/detect_issues.py \
  --input_csv "$RUN_ROOT/data/merged_dataset.csv" \
  --output_dir "$RUN_ROOT/reports"

python skills/data_quality/scripts/fix_data.py \
  --input_csv "$RUN_ROOT/data/merged_dataset.csv" \
  --output_csv "$RUN_ROOT/data/cleaned_dataset.csv" \
  --strategy smart

python skills/data_quality/scripts/compare_datasets.py \
  --before_csv "$RUN_ROOT/data/merged_dataset.csv" \
  --after_csv "$RUN_ROOT/data/cleaned_dataset.csv" \
  --output_report "$RUN_ROOT/reports/quality_comparison.md" \
  --output_html "$RUN_ROOT/reports/quality_comparison.html"

python skills/data_quality/scripts/save_strategy_justification.py \
  --strategy "smart" \
  --rationale "Selected after reviewing missing values/outlier trade-off with user." \
  --output "$RUN_ROOT/reports/strategy_justification.md"
```

7. Annotation + HITL:
```bash
python skills/annotation/scripts/auto_label.py \
  --input_csv "$RUN_ROOT/data/cleaned_dataset.csv" \
  --output_csv "$RUN_ROOT/data/auto_labeled_dataset.csv" \
  --column price \
  --rules '[{"type":"threshold","op":"<","val":10000,"label":"Budget","conf":0.9},{"type":"range","min":10000,"max":30000,"label":"Standard","conf":0.85},{"type":"range","min":30000,"max":60000,"label":"Premium","conf":0.75},{"type":"threshold","op":">=","val":60000,"label":"Luxury","conf":0.65}]'

python skills/annotation/scripts/build_review_queue.py \
  --input_csv "$RUN_ROOT/data/auto_labeled_dataset.csv" \
  --output_csv "$RUN_ROOT/data/review_queue.csv" \
  --threshold 0.7
```

После ручной правки `review_queue.csv`:
```bash
python skills/annotation/scripts/merge_reviewed.py \
  --auto_labeled_csv "$RUN_ROOT/data/auto_labeled_dataset.csv" \
  --reviewed_csv "$RUN_ROOT/data/review_queue.csv" \
  --output_csv "$RUN_ROOT/data/labeled_dataset.csv"

python skills/annotation/scripts/generate_spec.py \
  --input_csv "$RUN_ROOT/data/labeled_dataset.csv" \
  --task_desc "Car price segment labeling" \
  --output "$RUN_ROOT/labeling/annotation_spec.md"

python skills/annotation/scripts/check_quality.py \
  --input_csv "$RUN_ROOT/data/labeled_dataset.csv" \
  --output "$RUN_ROOT/reports/annotation_metrics.json"

python skills/annotation/scripts/export_to_labelstudio.py \
  --input_csv "$RUN_ROOT/data/labeled_dataset.csv" \
  --display_cols '["model","year","price","mileage","fuel_type","transmission"]' \
  --output_json "$RUN_ROOT/labeling/labelstudio_import.json"
```

8. Active Learning:
```bash
python skills/active_learning/scripts/run_experiment.py \
  --labeled_csv "$RUN_ROOT/data/labeled_dataset.csv" \
  --reports_dir "$RUN_ROOT/reports" \
  --models_dir "$RUN_ROOT/models" \
  --target_col auto_label \
  --n_start 50 \
  --iterations 5 \
  --batch_size 20
```

9. Финальный отчет (5 разделов):
```bash
python skills/orchestrator/scripts/build_final_report.py --run_root "$RUN_ROOT"
```

