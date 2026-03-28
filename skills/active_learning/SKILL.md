# Skill: Active Learning

## Goal
Compare `entropy` vs `random` strategies and quantify labeling savings.

## Minimum deliverables
- `reports/al_history_random.json`
- `reports/al_history_entropy.json`
- `reports/al_learning_curves.png`
- `reports/al_learning_curves.html`
- `reports/al_conclusion.txt`
- model artifacts in `models/`

## Core scripts
- `scripts/train_model.py`
- `scripts/evaluate_model.py`
- `scripts/query_samples.py`
- `scripts/run_experiment.py`
- `scripts/generate_report.py`

## Required experiment setup
- Stratified split into train/test.
- Start labeled subset `N_START` and fixed batch expansion.
- 5+ iterations per strategy.
- Final conclusion must include sample savings against random baseline.
