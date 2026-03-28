from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def generate_spec(df: pd.DataFrame, task_desc: str, class_defs: dict[str, str], output_path: Path, label_col: str = "auto_label") -> Path:
    lines = [
        f"# Annotation Specification: {task_desc}\n\n",
        "## 1. Task Definition\n",
        f"{task_desc}\n\n",
        "## 2. Label Definitions\n",
    ]

    for cls, desc in class_defs.items():
        lines.append(f"- **{cls}**: {desc}\n")

    lines.append("\n## 3. Data Examples\n")
    for cls in class_defs:
        lines.append(f"\n### Class: {cls}\n")
        rows = df[df[label_col] == cls].head(3)
        if rows.empty:
            lines.append("- No examples in current sample\n")
            continue
        for _, row in rows.iterrows():
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if k != label_col)
            lines.append(f"- {text}\n")

    lines.append("\n## 4. Edge Cases\n")
    conf_col = "confidence" if "confidence" in df.columns else None
    for cls in class_defs:
        lines.append(f"\n### Class: {cls}\n")
        subset = df[df[label_col] == cls]
        if conf_col:
            subset = subset.sort_values(conf_col).head(2)
        else:
            subset = subset.head(2)
        if subset.empty:
            lines.append("- No borderline examples found in current sample\n")
            continue
        for _, row in subset.iterrows():
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if k != label_col)
            lines.append(f"- borderline: {text}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annotation specification markdown")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--task_desc", required=True)
    parser.add_argument("--class_defs", default="{}", help="JSON dict label->description")
    parser.add_argument("--label_col", default="auto_label")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    class_defs = json.loads(args.class_defs)
    if not class_defs:
        labels = sorted(df[args.label_col].dropna().astype(str).unique().tolist())
        class_defs = {label: "auto-generated" for label in labels}

    out = generate_spec(df, args.task_desc, class_defs, Path(args.output), label_col=args.label_col)
    print(out)


if __name__ == "__main__":
    main()
