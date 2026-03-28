from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def export_labelstudio(
    df: pd.DataFrame,
    output_path: Path,
    display_cols: list[str],
    label_col: str = "auto_label",
    confidence_col: str = "confidence",
) -> Path:
    tasks = []
    safe_display = [c for c in display_cols if c in df.columns]

    for _, row in df.iterrows():
        text_block = "\n".join(f"{col}: {row[col]}" for col in safe_display)
        label = str(row.get(label_col, "Unknown"))
        score = float(row.get(confidence_col, 0.5)) if confidence_col in df.columns else 0.5

        tasks.append(
            {
                "data": {"text": text_block},
                "predictions": [
                    {
                        "model_version": "Codex-AutoLabel-v1",
                        "score": score,
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [label]},
                                "score": score,
                            }
                        ],
                    }
                ],
                "annotations": [],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset into LabelStudio import JSON")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--display_cols", required=True, help="JSON list of columns displayed in LabelStudio text field")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--label_col", default="auto_label")
    parser.add_argument("--confidence_col", default="confidence")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    display_cols = json.loads(args.display_cols)
    out = export_labelstudio(df, Path(args.output_json), display_cols, args.label_col, args.confidence_col)
    print(out)


if __name__ == "__main__":
    main()
