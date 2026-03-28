from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


SOFTMAX_EPS = 1e-12


def _probabilities(pipeline, X: pd.DataFrame) -> np.ndarray:
    model = pipeline.named_steps["model"]
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    if hasattr(model, "decision_function"):
        logits = model.decision_function(pipeline.named_steps["prep"].transform(X))
        if logits.ndim == 1:
            logits = np.vstack([-logits, logits]).T
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        proba = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), SOFTMAX_EPS, None)
        return proba
    raise RuntimeError("Model does not expose predict_proba or decision_function")


def query_samples(pool_csv: Path, model_path: Path, strategy: str, batch_size: int, target_col: str) -> np.ndarray:
    pool = pd.read_csv(pool_csv).reset_index(drop=True)
    if len(pool) == 0:
        return np.array([], dtype=int)

    if strategy == "random":
        rng = np.random.default_rng(42 + len(pool))
        return rng.choice(len(pool), size=min(batch_size, len(pool)), replace=False)

    with model_path.open("rb") as f:
        payload = pickle.load(f)
    pipeline = payload["pipeline"]
    drop_cols = payload.get("drop_cols", [])
    numeric_cols = payload.get("numeric_cols", [])
    categorical_cols = payload.get("categorical_cols", [])

    X = pool.drop(columns=[target_col], errors="ignore")
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    proba = _probabilities(pipeline, X)

    if strategy == "entropy":
        scores = -(proba * np.log(np.clip(proba, SOFTMAX_EPS, 1.0))).sum(axis=1)
    elif strategy == "margin":
        top2 = -np.sort(-proba, axis=1)[:, :2]
        scores = 1.0 - (top2[:, 0] - top2[:, 1])
    else:
        raise ValueError("Strategy must be one of: entropy, margin, random")

    idx = np.argsort(scores)[-min(batch_size, len(pool)):]
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Query most informative samples from unlabeled pool")
    parser.add_argument("--pool_csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--strategy", required=True, choices=["entropy", "margin", "random"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--target_col", default="auto_label")
    parser.add_argument("--output_indices", required=True)
    args = parser.parse_args()

    idx = query_samples(Path(args.pool_csv), Path(args.model_path), args.strategy, args.batch_size, args.target_col)
    out = Path(args.output_indices)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, idx)
    print(out)


if __name__ == "__main__":
    main()
