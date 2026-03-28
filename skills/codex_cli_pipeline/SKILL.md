# Skill: Codex CLI Universal Pipeline

## Purpose
Run the full reusable data pipeline **through Codex CLI interaction** (not as a standalone app), with explicit HITL checkpoints.

## Activation
Use this exact style in Codex CLI:

`Активируй @skills/codex_cli_pipeline/SKILL.md и проведи меня по пайплайну для темы: <topic>`

## Agent behavior (required)
1. Ask user for `topic` if missing.
2. Create run folder and keep using the same `RUN_ROOT` for all stages.
3. Execute each stage command below through terminal.
4. After each stage, summarize artifacts and ask user confirmation before continuing.
5. For annotation, enforce two-step HITL flow (`queue_only` then `merge_only`).

## Commands (stage-by-stage)

### 0) Init run
```bash
RUN_ROOT=$(python skills/orchestrator/scripts/init_run.py --task_name "<topic>")
```

### 1) Data collection (auto-discovery for the topic)
```bash
python skills/orchestrator/scripts/run_pipeline.py \
  --run_root "$RUN_ROOT" \
  --topic "<topic>" \
  --only_stage data_collection \
  --auto_confirm
```

### 2) Data quality
```bash
python skills/orchestrator/scripts/run_pipeline.py \
  --run_root "$RUN_ROOT" \
  --topic "<topic>" \
  --only_stage data_quality \
  --auto_confirm
```

### 3) Annotation queue generation (HITL step 1)
```bash
python skills/orchestrator/scripts/run_pipeline.py \
  --run_root "$RUN_ROOT" \
  --topic "<topic>" \
  --only_stage annotation \
  --annotation_mode queue_only \
  --auto_confirm
```

Then ask user to manually edit:
- `$RUN_ROOT/data/review_queue.csv`

### 4) Annotation merge/finalize (HITL step 2)
```bash
python skills/orchestrator/scripts/run_pipeline.py \
  --run_root "$RUN_ROOT" \
  --topic "<topic>" \
  --only_stage annotation \
  --annotation_mode merge_only \
  --auto_confirm
```

### 5) Active learning
```bash
python skills/orchestrator/scripts/run_pipeline.py \
  --run_root "$RUN_ROOT" \
  --topic "<topic>" \
  --only_stage active_learning \
  --auto_confirm
```

### 6) Final report
```bash
python skills/orchestrator/scripts/run_pipeline.py \
  --run_root "$RUN_ROOT" \
  --topic "<topic>" \
  --only_stage final_report \
  --auto_confirm
```

## Output checklist
- `reports/eda_report.html`
- `reports/quality_report.{md,html,json}`
- `reports/quality_comparison.{md,html}`
- `reports/strategy_justification.md`
- `data/review_queue.csv`
- `data/labeled_dataset.csv`
- `labeling/annotation_spec.md`
- `labeling/labelstudio_import.json`
- `reports/al_history_random.json`
- `reports/al_history_entropy.json`
- `reports/al_learning_curves.{png,html}`
- `reports/al_conclusion.txt`
- `README.md` (final 5-section report)
