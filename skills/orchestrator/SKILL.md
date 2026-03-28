# Skill: Orchestrator (Codex CLI)

## Goal
Coordinate full end-to-end pipeline with Human-in-the-Loop checkpoints and isolated run artifacts.

## Workflow
1. Initialize run folder:
```bash
python skills/orchestrator/scripts/init_run.py --task_name "<task-name>"
```
2. Run `data_collection` skill and stop for user confirmation.
3. Run `data_quality` skill and stop for user confirmation.
4. Run `annotation` skill and enforce manual `review_queue.csv` step.
5. Run `active_learning` skill and show savings summary.
6. Build final report with 5 required sections:
```bash
python skills/orchestrator/scripts/build_final_report.py --run_root "<run_root>"
```

## Hard requirements
- All outputs must stay inside one `data/raw/<task>_<timestamp>` folder.
- Ask for explicit user confirmation before each stage transition.
- Main communication language: Russian.

## Single-Entry Mode (Optional)
Run full pipeline with one command:
```bash
python skills/orchestrator/scripts/run_pipeline.py --topic "<your-domain>" --auto_confirm
```

For strict HITL flow (manual queue pause enabled):
```bash
python skills/orchestrator/scripts/run_pipeline.py --topic "<your-domain>"
```

For demo/non-interactive run (skip manual pause):
```bash
python skills/orchestrator/scripts/run_pipeline.py --topic "<your-domain>" --auto_confirm --no_review_pause
```

Stage-scoped execution remains available:
```bash
python skills/orchestrator/scripts/run_pipeline.py --run_root "<existing_run_root>" --from_stage annotation --to_stage final_report
```
