from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1500)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def train_model(labeled_csv: Path, model_path: Path, target_col: str, drop_cols: list[str] | None = None) -> Path:
    drop_cols = drop_cols or []

    df = pd.read_csv(labeled_csv)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    keep_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    keep_df = keep_df.dropna(subset=[target_col]).reset_index(drop=True)

    y = keep_df[target_col].astype(str)
    X = keep_df.drop(columns=[target_col])

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
            continue
        numeric_view = pd.to_numeric(X[col], errors="coerce")
        ratio = float(numeric_view.notna().mean())
        if ratio >= 0.95:
            X[col] = numeric_view
            numeric_cols.append(col)
        else:
            X[col] = X[col].astype(str)
            categorical_cols.append(col)

    if y.nunique() < 2:
        pipeline = Pipeline(
            steps=[("prep", ColumnTransformer(transformers=[], remainder="drop")), ("model", DummyClassifier(strategy="most_frequent"))]
        )
    else:
        pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X, y)

    payload = {
        "pipeline": pipeline,
        "target_col": target_col,
        "drop_cols": drop_cols,
        "feature_cols": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(payload, f)

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AL base model")
    parser.add_argument("--labeled_csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--drop_cols", default="[]", help="JSON list of non-feature columns")
    args = parser.parse_args()

    out = train_model(
        labeled_csv=Path(args.labeled_csv),
        model_path=Path(args.model_path),
        target_col=args.target_col,
        drop_cols=json.loads(args.drop_cols),
    )
    print(out)


if __name__ == "__main__":
    main()
