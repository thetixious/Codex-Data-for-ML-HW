from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


POS_WORDS = {"good", "great", "excellent", "awesome", "love", "best", "нрав", "класс", "отлич"}
NEG_WORDS = {"bad", "awful", "terrible", "hate", "worst", "poor", "ужас", "плох", "ненав"}


@dataclass
class QualityMetrics:
    kappa: float | None
    label_dist: dict[str, float]
    confidence_mean: float


class AnnotationAgent:
    def __init__(self, modality: str = "text") -> None:
        self.modality = modality

    def auto_label(self, df: pd.DataFrame, modality: str | None = None) -> pd.DataFrame:
        mod = (modality or self.modality).lower()
        out = df.copy().reset_index(drop=True)
        out["__row_id"] = out.index

        if mod != "text":
            raise NotImplementedError("Current implementation supports text modality")

        if "label" in out.columns:
            out["auto_label"] = out["label"].astype(str)
            out["confidence"] = 0.95
            out["reason"] = "copied_existing_label"
            return out

        text_col = self._find_text_col(out)
        if text_col is None:
            raise ValueError("No text column found for text auto-labeling")

        labels = []
        confidences = []
        reasons = []

        for text in out[text_col].fillna("").astype(str):
            label, conf, reason = self._label_text(text)
            labels.append(label)
            confidences.append(conf)
            reasons.append(reason)

        out["auto_label"] = labels
        out["confidence"] = confidences
        out["reason"] = reasons
        return out

    def generate_spec(self, df: pd.DataFrame, task: str) -> str:
        label_col = "auto_label" if "auto_label" in df.columns else "label"
        labels = sorted(df[label_col].dropna().astype(str).unique().tolist())

        lines = [
            f"# Annotation Spec: {task}\n\n",
            "## Task\n",
            f"{task}\n\n",
            "## Classes\n",
        ]

        for label in labels:
            lines.append(f"- **{label}**: class definition placeholder\n")

        lines.append("\n## Examples (3+ per class)\n")
        lines.append("\n## Edge Cases\n")

        for label in labels:
            lines.append(f"\n### {label}\n")
            subset = df[df[label_col].astype(str) == label].head(3)
            text_col = self._find_text_col(subset)
            if text_col is None or subset.empty:
                lines.append("- no examples\n")
            else:
                for _, row in subset.iterrows():
                    lines.append(f"- {str(row[text_col])[:280]}\n")

            lines.append("- borderline case 1\n")
            lines.append("- borderline case 2\n")

        return "".join(lines)

    def check_quality(self, df_labeled: pd.DataFrame, human_col: str | None = None) -> dict[str, Any]:
        auto_col = "auto_label" if "auto_label" in df_labeled.columns else "label"

        kappa = None
        if human_col and human_col in df_labeled.columns:
            kappa = float(cohen_kappa_score(df_labeled[auto_col], df_labeled[human_col]))

        dist = df_labeled[auto_col].value_counts(normalize=True)
        label_dist = {str(k): float(v) for k, v in dist.to_dict().items()}

        conf = df_labeled["confidence"] if "confidence" in df_labeled.columns else pd.Series([0.0] * len(df_labeled))
        metrics = {
            "kappa": kappa,
            "label_dist": label_dist,
            "confidence_mean": float(conf.mean()),
            "agreement": None,
        }

        if human_col and human_col in df_labeled.columns:
            metrics["agreement"] = float((df_labeled[auto_col].astype(str) == df_labeled[human_col].astype(str)).mean())

        return metrics

    def export_to_labelstudio(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        text_col = self._find_text_col(df)
        if text_col is None:
            text_col = df.columns[0]

        label_col = "auto_label" if "auto_label" in df.columns else "label"
        conf_col = "confidence" if "confidence" in df.columns else None

        tasks = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            label = str(row.get(label_col, "Unknown"))
            score = float(row.get(conf_col, 0.5)) if conf_col else 0.5

            tasks.append(
                {
                    "data": {"text": text},
                    "predictions": [
                        {
                            "model_version": "AnnotationAgent-v1",
                            "score": score,
                            "result": [
                                {
                                    "type": "choices",
                                    "value": {"choices": [label]},
                                    "from_name": "label",
                                    "to_name": "text",
                                    "score": score,
                                }
                            ],
                        }
                    ],
                    "annotations": [],
                }
            )
        return tasks

    @staticmethod
    def save_spec(spec_text: str, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spec_text, encoding="utf-8")
        return path

    @staticmethod
    def save_labelstudio(tasks: list[dict[str, Any]], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _find_text_col(df: pd.DataFrame) -> str | None:
        for col in df.columns:
            norm = "".join(ch.lower() if ch.isalnum() else "_" for ch in col)
            if any(key in norm for key in ["text", "comment", "review", "content", "title", "question"]):
                return col
        for col in df.columns:
            if df[col].dtype == object:
                return col
        return None

    @staticmethod
    def _label_text(text: str) -> tuple[str, float, str]:
        t = text.lower()
        pos = sum(1 for w in POS_WORDS if w in t)
        neg = sum(1 for w in NEG_WORDS if w in t)

        if pos > neg:
            return "positive", min(0.95, 0.6 + 0.1 * (pos - neg)), "lexicon_positive"
        if neg > pos:
            return "negative", min(0.95, 0.6 + 0.1 * (neg - pos)), "lexicon_negative"
        return "neutral", 0.55, "lexicon_neutral"


__all__ = ["AnnotationAgent", "QualityMetrics"]
