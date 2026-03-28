# Final Pipeline Report: mortage risk predection

## 1. Task and Dataset
- Task: Discover and collect datasets for a user-selected ML domain,
then run quality audit, annotation with HITL, and active learning.

- Labeled rows: 588
- Columns: 44
- Classes/labels: tier_1, tier_2, tier_3, tier_4

## 2. Stage Decisions
- Data collection: multi-source ingestion + schema unification completed.
- Data quality: issue types detected = 2, strategy selected by user at checkpoint.
- Annotation: auto-labeling done using explicit rules and confidence scoring.
- Active learning: entropy vs random comparison completed.

## 3. Human-in-the-Loop Checkpoint
- Rows sent to manual review (confidence threshold): 149
- Reviewed queue merged back into final labeled dataset.

## 4. Metrics by Stage
- Quality: duplicates = 0, missing columns with gaps = 38.
- Annotation: kappa = 0.0, mean confidence = 0.7746598639455782.
- AL random final: accuracy = 0.847457627118644, f1 = 0.829635220917006.
- AL entropy final: accuracy = 0.9152542372881356, f1 = 0.9196670538133954.
- AL conclusion: Using Active Learning (entropy) reached random-baseline quality with 100 fewer labeled examples (66.7%).

## 5. Retrospective
- What worked: isolated runs, stage checkpoints, reproducible artifacts, and AL comparison history.
- What did not: performance may vary by label quality and class imbalance in initial labeled subset.
- What to improve next: stronger base model, richer annotation policy, and larger validated seed set.
