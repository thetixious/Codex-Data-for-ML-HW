from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(test_csv: Path, model_path: Path, target_col: str) -> dict[str, float]:
    df = pd.read_csv(test_csv)

    with model_path.open("rb") as f:
        payload = pickle.load(f)

    pipeline = payload["pipeline"]
    drop_cols = payload.get("drop_cols", [])
    numeric_cols = payload.get("numeric_cols", [])
    categorical_cols = payload.get("categorical_cols", [])

    work = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    work = work.dropna(subset=[target_col]).reset_index(drop=True)

    y_true = work[target_col].astype(str)
    X = work.drop(columns=[target_col])
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    y_pred = pipeline.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained AL model")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    metrics = evaluate_model(Path(args.test_csv), Path(args.model_path), args.target_col)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
