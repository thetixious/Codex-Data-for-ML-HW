from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def calculate_metrics(df: pd.DataFrame, auto_col: str, human_col: str | None, confidence_col: str = "confidence") -> dict:
    metrics: dict[str, object] = {}

    if human_col and human_col in df.columns:
        kappa = float(cohen_kappa_score(df[auto_col], df[human_col]))
        metrics["kappa"] = None if math.isnan(kappa) else kappa
        metrics["agreement"] = float((df[auto_col].astype(str) == df[human_col].astype(str)).mean())
    else:
        metrics["kappa"] = None
        metrics["agreement"] = None

    dist = df[auto_col].value_counts(normalize=True)
    metrics["label_distribution"] = {str(k): float(v) for k, v in dist.to_dict().items()}

    if confidence_col in df.columns:
        conf = df[confidence_col]
        metrics["confidence_mean"] = float(conf.mean())
        metrics["confidence_median"] = float(conf.median())
        metrics["low_confidence_lt_0_7"] = int((conf < 0.7).sum())

    metrics["rows"] = int(len(df))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute labeling quality metrics")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--auto_col", default="auto_label")
    parser.add_argument("--human_col", default=None)
    parser.add_argument("--confidence_col", default="confidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    metrics = calculate_metrics(df, args.auto_col, args.human_col, args.confidence_col)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
