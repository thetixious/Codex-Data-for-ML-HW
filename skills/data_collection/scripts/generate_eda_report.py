from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.html_report import build_html_page, save_html
from utils.logger import get_logger

logger = get_logger(__name__)


MAX_NUMERIC_PLOTS = 4
MAX_CATEGORICAL_PLOTS = 4


def _card(title: str, body: str) -> str:
    return f"<section class='card'><h2>{title}</h2>{body}</section>"


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def generate_eda_report(input_csv: Path, output_html: Path, task_description: str = "") -> Path:
    df = pd.read_csv(input_csv)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    sections: list[str] = []

    overview = [
        f"<p><b>Generated:</b> {datetime.now().isoformat(timespec='seconds')}</p>",
        f"<p><b>Source file:</b> {input_csv}</p>",
        f"<p><b>Rows:</b> {len(df):,} | <b>Columns:</b> {len(df.columns)}</p>",
    ]
    if task_description.strip():
        overview.append(f"<p><b>Task:</b> {task_description}</p>")
    sections.append(_card("Overview", "".join(overview)))

    head_html = df.head(20).to_html(index=False)
    sections.append(_card("Sample (Top 20 rows)", head_html))

    missing = df.isna().sum().sort_values(ascending=False)
    missing_df = pd.DataFrame({"column": missing.index, "missing": missing.values})
    fig_missing = px.bar(missing_df, x="column", y="missing", title="Missing values by column")
    sections.append(_card("Missing Values", _fig_html(fig_missing)))

    if numeric_cols:
        desc = df[numeric_cols].describe().T.round(4)
        sections.append(_card("Numeric Summary", desc.to_html()))

        for col in numeric_cols[:MAX_NUMERIC_PLOTS]:
            fig_hist = px.histogram(df, x=col, marginal="box", nbins=40, title=f"Distribution: {col}")
            sections.append(_card(f"Distribution - {col}", _fig_html(fig_hist)))

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap")
            sections.append(_card("Correlation", _fig_html(fig_corr)))

    if categorical_cols:
        for col in categorical_cols[:MAX_CATEGORICAL_PLOTS]:
            top = df[col].astype(str).value_counts(dropna=False).head(20).reset_index()
            top.columns = [col, "count"]
            fig_cat = px.bar(top, x=col, y="count", title=f"Top categories: {col}")
            sections.append(_card(f"Categorical - {col}", _fig_html(fig_cat)))

    insights: list[str] = []
    if numeric_cols:
        skewed = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) > 10 and abs(s.skew()) > 1.0:
                skewed.append(col)
        if skewed:
            insights.append(f"<li>Highly skewed numeric features: {', '.join(skewed[:8])}</li>")
    if categorical_cols:
        high_card = [c for c in categorical_cols if df[c].nunique(dropna=True) > 50]
        if high_card:
            insights.append(f"<li>High-cardinality categorical features: {', '.join(high_card[:8])}</li>")
    if not insights:
        insights.append("<li>No critical anomalies were auto-detected at EDA stage.</li>")

    sections.append(_card("Auto Insights", f"<ul>{''.join(insights)}</ul>"))

    html = build_html_page("EDA Report", sections)
    save_html(output_html, html)

    logger.info("EDA report saved -> %s", output_html)
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive HTML EDA report")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--task_description", default="")
    args = parser.parse_args()

    out = generate_eda_report(Path(args.input_csv), Path(args.output_html), args.task_description)
    print(out)


if __name__ == "__main__":
    main()
