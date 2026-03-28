# Data Card

- Task/domain: sentiment_smoke
- Rows: 8540
- Columns: __row_id, text, audio, image, label, source, collected_at, auto_label, confidence, reason, human_label
- Class distribution: {"1": 4265, "0": 4265, "unlabeled": 10}
- Source distribution: {"hf:rotten_tomatoes": 8530, "zenodo_api": 10}

## Schema

|              | dtype   |
|:-------------|:--------|
| __row_id     | int64   |
| text         | str     |
| audio        | str     |
| image        | str     |
| label        | str     |
| source       | str     |
| collected_at | str     |
| auto_label   | str     |
| confidence   | float64 |
| reason       | str     |
| human_label  | str     |
