from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    missing: dict[str, dict[str, float]]
    duplicates: int
    outliers: dict[str, dict[str, float]]
    imbalance: dict[str, Any] | None


class DataQualityAgent:
    def detect_issues(self, df: pd.DataFrame, label_col: str | None = None) -> dict[str, Any]:
        missing = {}
        miss = df.isna().sum()
        for col, cnt in miss.items():
            if cnt > 0:
                missing[col] = {"count": int(cnt), "pct": float(100.0 * cnt / max(1, len(df)))}

        duplicates = int(df.duplicated().sum())

        outliers = {}
        for col in df.select_dtypes(include=np.number).columns:
            s = df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((s < low) | (s > high)).sum())
            if n > 0:
                outliers[col] = {
                    "count": n,
                    "pct": float(100.0 * n / max(1, len(s))),
                    "lower_bound": float(low),
                    "upper_bound": float(high),
                }

        imbalance = None
        if label_col and label_col in df.columns:
            vc = df[label_col].value_counts(dropna=False)
            if len(vc) > 1:
                imbalance = {
                    "col": label_col,
                    "counts": {str(k): int(v) for k, v in vc.to_dict().items()},
                    "ratio": float(vc.max() / max(1, vc.min())),
                }

        return {
            "missing": missing,
            "duplicates": duplicates,
            "outliers": outliers,
            "imbalance": imbalance,
        }

    def fix(self, df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        out = df.copy()

        dup_policy = strategy.get("duplicates", "drop")
        if dup_policy == "drop":
            out = out.drop_duplicates().reset_index(drop=True)

        miss_policy = strategy.get("missing", "median")
        out = self._fix_missing(out, miss_policy)

        out_policy = strategy.get("outliers", "clip_iqr")
        out = self._fix_outliers(out, out_policy)

        return out.reset_index(drop=True)

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        before_report = self.detect_issues(df_before)
        after_report = self.detect_issues(df_after)

        before = {
            "Rows": int(len(df_before)),
            "Missing values": int(df_before.isna().sum().sum()),
            "Duplicates": int(before_report["duplicates"]),
            "Outlier cells (IQR)": int(sum(v["count"] for v in before_report["outliers"].values())),
        }
        after = {
            "Rows": int(len(df_after)),
            "Missing values": int(df_after.isna().sum().sum()),
            "Duplicates": int(after_report["duplicates"]),
            "Outlier cells (IQR)": int(sum(v["count"] for v in after_report["outliers"].values())),
        }

        rows = []
        for metric in before:
            b, a = before[metric], after[metric]
            delta = a - b
            pct = (delta / b * 100.0) if b != 0 else np.nan
            rows.append({"Metric": metric, "Before": b, "After": a, "Delta": delta, "Change": pct})

        return pd.DataFrame(rows)

    @staticmethod
    def _fix_missing(df: pd.DataFrame, policy: str) -> pd.DataFrame:
        out = df.copy()
        num_cols = out.select_dtypes(include=np.number).columns
        cat_cols = [c for c in out.columns if c not in num_cols]

        if policy == "drop":
            return out.dropna()
        if policy == "ffill":
            return out.fillna(method="ffill")
        if policy.startswith("constant:"):
            return out.fillna(policy.split(":", 1)[1])

        for col in num_cols:
            if policy == "mean":
                out[col] = out[col].fillna(out[col].mean())
            elif policy == "median":
                out[col] = out[col].fillna(out[col].median())
            elif policy == "mode":
                mode = out[col].mode(dropna=True)
                if not mode.empty:
                    out[col] = out[col].fillna(mode.iloc[0])

        for col in cat_cols:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])

        return out

    @staticmethod
    def _fix_outliers(df: pd.DataFrame, policy: str) -> pd.DataFrame:
        out = df.copy()
        num_cols = out.select_dtypes(include=np.number).columns

        if policy == "none":
            return out

        for col in num_cols:
            s = out[col]
            if s.dropna().empty:
                continue

            if policy in {"clip_iqr", "drop"}:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            elif policy == "clip_zscore":
                m, std = s.mean(), s.std(ddof=0)
                if std == 0:
                    continue
                low, high = m - 3 * std, m + 3 * std
            else:
                continue

            if policy.startswith("clip"):
                out[col] = out[col].clip(lower=low, upper=high)
            elif policy == "drop":
                out = out[(out[col] >= low) & (out[col] <= high)]

        return out


__all__ = ["DataQualityAgent", "QualityReport"]
