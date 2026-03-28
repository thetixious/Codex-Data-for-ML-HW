from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from skills.active_learning.scripts.evaluate_model import evaluate_model
from skills.active_learning.scripts.generate_report import generate_reports
from skills.active_learning.scripts.query_samples import query_samples
from skills.active_learning.scripts.train_model import train_model


def _stratify_or_none(y: pd.Series, n_samples: int, test_size: float | int | None) -> pd.Series | None:
    y_str = y.astype(str)
    value_counts = y_str.value_counts(dropna=False)
    if len(value_counts) <= 1:
        return None
    if int(value_counts.min()) < 2:
        return None

    if isinstance(test_size, float):
        n_test = int(round(n_samples * test_size))
    elif isinstance(test_size, int):
        n_test = int(test_size)
    else:
        n_test = 0

    n_test = max(1, min(n_samples - 1, n_test))
    n_train = n_samples - n_test
    n_classes = int(value_counts.shape[0])
    if n_test < n_classes or n_train < n_classes:
        return None
    return y_str


def _split_with_fallback(
    df: pd.DataFrame,
    *,
    target_col: str,
    random_state: int,
    test_size: float | int | None = None,
    train_size: float | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    if test_size is not None:
        stratify = _stratify_or_none(df[target_col], len(df), test_size)

    try:
        return train_test_split(
            df,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            df,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            stratify=None,
        )


def _run_strategy(
    strategy: str,
    labeled_start: pd.DataFrame,
    pool_start: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    n_iterations: int,
    batch_size: int,
    model_dir: Path,
) -> list[dict]:
    labeled = labeled_start.copy().reset_index(drop=True)
    pool = pool_start.copy().reset_index(drop=True)

    history: list[dict] = []
    for it in range(n_iterations + 1):
        labeled_path = model_dir / f"tmp_labeled_{strategy}_{it}.csv"
        model_path = model_dir / f"model_{strategy}_v{it}.pkl"
        test_path = model_dir / "tmp_test.csv"

        labeled.to_csv(labeled_path, index=False)
        test_df.to_csv(test_path, index=False)

        train_model(labeled_path, model_path, target_col=target_col, drop_cols=drop_cols)
        metrics = evaluate_model(test_path, model_path, target_col=target_col)
        history.append(
            {
                "iteration": it,
                "n_labeled": int(len(labeled)),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
            }
        )

        if it == n_iterations or len(pool) == 0:
            continue

        pool_path = model_dir / f"tmp_pool_{strategy}_{it}.csv"
        pool.to_csv(pool_path, index=False)
        idx = query_samples(pool_path, model_path, strategy=strategy, batch_size=batch_size, target_col=target_col)
        idx = sorted(set(int(i) for i in idx.tolist()))
        if not idx:
            continue

        selected = pool.iloc[idx].copy()
        remaining = pool.drop(index=idx).reset_index(drop=True)

        labeled = pd.concat([labeled, selected], ignore_index=True).reset_index(drop=True)
        pool = remaining

    return history


def run_experiment(
    labeled_csv: Path,
    reports_dir: Path,
    models_dir: Path,
    target_col: str,
    drop_cols: list[str] | None,
    n_start: int,
    iterations: int,
    batch_size: int,
    test_size: float,
    random_state: int,
) -> None:
    drop_cols = drop_cols or []
    df = pd.read_csv(labeled_csv)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) <= n_start + 10:
        raise ValueError("Dataset too small for AL experiment with current n_start")

    train_df, test_df = _split_with_fallback(
        df,
        target_col=target_col,
        random_state=random_state,
        test_size=test_size,
    )

    labeled_start, pool_start = _split_with_fallback(
        train_df,
        target_col=target_col,
        random_state=random_state,
        train_size=n_start,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}
    for strategy in ["random", "entropy"]:
        history = _run_strategy(
            strategy=strategy,
            labeled_start=labeled_start,
            pool_start=pool_start,
            test_df=test_df,
            target_col=target_col,
            drop_cols=drop_cols,
            n_iterations=iterations,
            batch_size=batch_size,
            model_dir=models_dir,
        )
        results[strategy] = history
        (reports_dir / f"al_history_{strategy}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    generate_reports(
        history_files=[reports_dir / "al_history_random.json", reports_dir / "al_history_entropy.json"],
        labels=["random", "entropy"],
        output_img=reports_dir / "al_learning_curves.png",
        output_html=reports_dir / "al_learning_curves.html",
        output_conclusion=reports_dir / "al_conclusion.txt",
        metric="f1",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Active Learning experiment")
    parser.add_argument("--labeled_csv", required=True)
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--target_col", default="auto_label")
    parser.add_argument("--drop_cols", default="[]", help="JSON list of columns to exclude from AL features")
    parser.add_argument("--n_start", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        labeled_csv=Path(args.labeled_csv),
        reports_dir=Path(args.reports_dir),
        models_dir=Path(args.models_dir),
        target_col=args.target_col,
        drop_cols=json.loads(args.drop_cols),
        n_start=args.n_start,
        iterations=args.iterations,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(Path(args.reports_dir) / "al_conclusion.txt")


if __name__ == "__main__":
    main()
