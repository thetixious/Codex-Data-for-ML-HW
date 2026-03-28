from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.html_report import build_html_page, save_html
from utils.logger import get_logger

logger = get_logger(__name__)


def _iqr_outliers(s: pd.Series) -> tuple[int, float, float]:
    s = s.dropna()
    if s.empty:
        return 0, float("nan"), float("nan")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n = int(((s < low) | (s > high)).sum())
    return n, float(low), float(high)


def detect_issues(input_csv: Path, output_dir: Path, label_col: str | None = None) -> dict:
    df = pd.read_csv(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = df.isna().sum().sort_values(ascending=False)
    missing_nonzero = missing[missing > 0]
    duplicates = int(df.duplicated().sum())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    outliers: dict[str, dict[str, float]] = {}
    for col in num_cols:
        count, low, high = _iqr_outliers(df[col])
        if count > 0:
            outliers[col] = {"count": count, "lower": low, "upper": high}

    imbalance = None
    if label_col and label_col in df.columns:
        vc = df[label_col].value_counts(dropna=False)
        if len(vc) > 1 and vc.min() > 0:
            ratio = float(vc.max() / vc.min())
            imbalance = {"counts": vc.to_dict(), "ratio": ratio}

    issue_types = 0
    issue_types += int(len(missing_nonzero) > 0)
    issue_types += int(duplicates > 0)
    issue_types += int(len(outliers) > 0)
    issue_types += int(bool(imbalance and imbalance["ratio"] > 1.5))

    report = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "missing": {k: int(v) for k, v in missing_nonzero.to_dict().items()},
        "duplicates": duplicates,
        "outliers": outliers,
        "imbalance": imbalance,
        "issue_types_detected": issue_types,
    }

    md_lines = [
        f"# Data Quality Report: {input_csv.name}\n\n",
        f"- Rows: **{df.shape[0]}**\n",
        f"- Columns: **{df.shape[1]}**\n",
        f"- Issue types detected: **{issue_types}**\n\n",
        "## Missing Values\n\n",
    ]
    if missing_nonzero.empty:
        md_lines.append("No missing values found.\n\n")
    else:
        md_lines.append(missing_nonzero.to_frame("missing").to_markdown() + "\n\n")

    md_lines.append("## Duplicates\n\n")
    md_lines.append(f"Duplicate rows: **{duplicates}**\n\n")

    md_lines.append("## Outliers (IQR)\n\n")
    if outliers:
        md_lines.append(pd.DataFrame(outliers).T.to_markdown() + "\n\n")
    else:
        md_lines.append("No numeric outliers detected by IQR.\n\n")

    if imbalance is not None:
        md_lines.append("## Class Balance\n\n")
        md_lines.append(f"Class ratio (max/min): **{imbalance['ratio']:.3f}**\n\n")
        md_lines.append(pd.Series(imbalance["counts"], name="count").to_markdown() + "\n\n")

    (output_dir / "quality_report.md").write_text("".join(md_lines), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    sections: list[str] = []
    sections.append(
        "<section class='card'><h2>Overview</h2>"
        f"<p><b>Rows:</b> {df.shape[0]} | <b>Columns:</b> {df.shape[1]} | <b>Issue types:</b> {issue_types}</p>"
        "</section>"
    )

    miss_df = missing.reset_index()
    miss_df.columns = ["column", "missing"]
    fig_missing = px.bar(miss_df, x="column", y="missing", title="Missing values")
    sections.append(f"<section class='card'><h2>Missing Values</h2>{fig_missing.to_html(full_html=False, include_plotlyjs='cdn')}</section>")

    if num_cols:
        outlier_counts = {c: _iqr_outliers(df[c])[0] for c in num_cols}
        oc_df = pd.DataFrame({"column": list(outlier_counts.keys()), "outliers": list(outlier_counts.values())})
        fig_out = px.bar(oc_df, x="column", y="outliers", title="Outlier count by numeric feature")
        sections.append(f"<section class='card'><h2>Outliers</h2>{fig_out.to_html(full_html=False, include_plotlyjs='cdn')}</section>")

    if imbalance is not None:
        bdf = pd.Series(imbalance["counts"], name="count").reset_index()
        bdf.columns = ["label", "count"]
        fig_bal = px.pie(bdf, names="label", values="count", title="Class distribution")
        sections.append(f"<section class='card'><h2>Class Balance</h2>{fig_bal.to_html(full_html=False, include_plotlyjs='cdn')}</section>")

    html = build_html_page("Data Quality Report", sections)
    save_html(output_dir / "quality_report.html", html)

    logger.info("Quality reports saved to %s", output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect quality issues in CSV")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_col", default=None)
    args = parser.parse_args()

    report = detect_issues(Path(args.input_csv), Path(args.output_dir), args.label_col)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
