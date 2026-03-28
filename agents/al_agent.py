from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


Strategy = Literal["entropy", "margin", "random"]


@dataclass
class IterationResult:
    iteration: int
    n_labeled: int
    accuracy: float
    f1: float
    strategy: str
    queried_indices: list[int]


class ActiveLearningAgent:
    def __init__(self, model: str = "logreg", random_state: int = 42) -> None:
        self.model_name = model
        self.random_state = random_state
        self.model = self._build_model()

    def _build_model(self) -> Pipeline:
        clf = LogisticRegression(max_iter=1200, random_state=self.random_state)
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)),
                ("clf", clf),
            ]
        )

    def fit(self, labeled_df: pd.DataFrame, text_col: str = "text", label_col: str = "auto_label") -> Pipeline:
        X = self._to_text(labeled_df, text_col=text_col)
        y = labeled_df[label_col].astype(str)
        self.model.fit(X, y)
        return self.model

    def query(self, pool: pd.DataFrame, strategy: Strategy = "entropy", n: int = 20, text_col: str = "text") -> list[int]:
        if len(pool) == 0:
            return []

        n = min(n, len(pool))
        if strategy == "random":
            rng = np.random.default_rng(self.random_state + len(pool))
            return sorted(rng.choice(len(pool), size=n, replace=False).tolist())

        X = self._to_text(pool, text_col=text_col)
        proba = self.model.predict_proba(X)

        if strategy == "entropy":
            scores = -(proba * np.log(np.clip(proba, 1e-12, 1.0))).sum(axis=1)
        else:  # margin
            top2 = -np.sort(-proba, axis=1)[:, :2]
            scores = 1.0 - (top2[:, 0] - top2[:, 1])

        idx = np.argsort(scores)[-n:]
        return sorted(idx.tolist())

    def evaluate(self, labeled_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "text", label_col: str = "auto_label") -> dict[str, float]:
        self.fit(labeled_df, text_col=text_col, label_col=label_col)
        X_test = self._to_text(test_df, text_col=text_col)
        y_true = test_df[label_col].astype(str)
        y_pred = self.model.predict(X_test)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        strategy: Strategy = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
        test_df: pd.DataFrame | None = None,
        text_col: str = "text",
        label_col: str = "auto_label",
    ) -> list[dict]:
        labeled = labeled_df.reset_index(drop=True).copy()
        pool = pool_df.reset_index(drop=True).copy()

        if test_df is None:
            test_df = labeled.sample(min(len(labeled), max(20, int(0.2 * len(labeled)))), random_state=self.random_state)

        history: list[dict] = []

        for it in range(n_iterations + 1):
            metrics = self.evaluate(labeled, test_df, text_col=text_col, label_col=label_col)

            queried: list[int] = []
            if it < n_iterations and len(pool) > 0:
                self.fit(labeled, text_col=text_col, label_col=label_col)
                queried = self.query(pool, strategy=strategy, n=batch_size, text_col=text_col)

                selected = pool.iloc[queried].copy()
                pool = pool.drop(index=queried).reset_index(drop=True)
                labeled = pd.concat([labeled, selected], ignore_index=True)

            row = IterationResult(
                iteration=it,
                n_labeled=int(len(labeled)),
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
                strategy=strategy,
                queried_indices=queried,
            )
            history.append(asdict(row))

        return history

    def report(self, history: list[dict], output_path: str | Path = "learning_curve.png") -> Path:
        if not history:
            raise ValueError("Empty history")

        df = pd.DataFrame(history)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(9, 5))
        for strat in sorted(df["strategy"].unique()):
            rows = df[df["strategy"] == strat].sort_values("n_labeled")
            plt.plot(rows["n_labeled"], rows["f1"], marker="o", label=f"{strat} (F1)")
        plt.xlabel("n_labeled")
        plt.ylabel("F1")
        plt.title("Active Learning: quality vs labeled samples")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()
        return out

    @staticmethod
    def compare(histories: dict[str, list[dict]], metric: str = "f1") -> dict[str, dict[str, float]]:
        if "random" not in histories or not histories["random"]:
            return {}

        target = histories["random"][-1][metric]
        n_random = histories["random"][-1]["n_labeled"]

        out: dict[str, dict[str, float]] = {}
        for strat, hist in histories.items():
            if strat == "random":
                continue
            reached = next((r for r in hist if r[metric] >= target), None)
            if reached is None:
                continue
            saved = n_random - reached["n_labeled"]
            out[strat] = {
                "target_metric": float(target),
                "n_random": int(n_random),
                "n_reached": int(reached["n_labeled"]),
                "saved": int(saved),
                "saved_pct": float(100.0 * saved / max(1, n_random)),
            }
        return out

    @staticmethod
    def _to_text(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
        if text_col in df.columns:
            return df[text_col].fillna("").astype(str)

        cols = [c for c in df.columns if c not in {"auto_label", "label", "source", "collected_at"}]
        if not cols:
            cols = list(df.columns)
        return df[cols].fillna("").astype(str).agg(" ".join, axis=1)


__all__ = ["ActiveLearningAgent", "IterationResult"]
