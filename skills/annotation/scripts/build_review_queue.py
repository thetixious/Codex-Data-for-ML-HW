from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_review_queue(input_csv: Path, output_csv: Path, threshold: float, confidence_col: str = "confidence") -> Path:
    df = pd.read_csv(input_csv)
    if confidence_col not in df.columns:
        raise KeyError(f"Confidence column '{confidence_col}' not found")

    queue = df[df[confidence_col] < threshold].copy().reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build low-confidence review queue")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--confidence_col", default="confidence")
    args = parser.parse_args()

    out = build_review_queue(Path(args.input_csv), Path(args.output_csv), args.threshold, args.confidence_col)
    print(out)


if __name__ == "__main__":
    main()
