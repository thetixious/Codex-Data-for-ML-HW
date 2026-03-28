from __future__ import annotations

import argparse
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


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def compare_data(before_csv: Path, after_csv: Path, output_report: Path, output_html: Path | None = None) -> Path:
    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)

    m_before = _metrics(before)
    m_after = _metrics(after)
    cmp_df = pd.DataFrame([{"dataset": "before", **m_before}, {"dataset": "after", **m_after}])

    lines = [
        "# Quality Comparison: Before vs After\n\n",
        "## Core Metrics\n\n",
        cmp_df.to_markdown(index=False),
        "\n\n",
    ]

    numeric_common = sorted(set(before.select_dtypes(include=np.number).columns) & set(after.select_dtypes(include=np.number).columns))
    if numeric_common:
        lines.append("## Numeric Statistics\n\n")
        for col in numeric_common:
            stats = pd.DataFrame({
                "before": before[col].describe()[["mean", "std", "min", "max"]],
                "after": after[col].describe()[["mean", "std", "min", "max"]],
            })
            lines.append(f"### {col}\n\n")
            lines.append(stats.to_markdown())
            lines.append("\n\n")

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text("".join(lines), encoding="utf-8")

    if output_html is not None:
        sections: list[str] = []
        sections.append("<section class='card'><h2>Core Metrics</h2>" + cmp_df.to_html(index=False) + "</section>")

        fig = px.bar(
            cmp_df.melt(id_vars=["dataset"], value_vars=["rows", "missing_cells", "duplicates"], var_name="metric", value_name="value"),
            x="metric",
            y="value",
            color="dataset",
            barmode="group",
            title="Before/After comparison",
        )
        sections.append(f"<section class='card'><h2>Metric Chart</h2>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</section>")

        html = build_html_page("Quality Comparison", sections)
        save_html(output_html, html)

    logger.info("Comparison report saved -> %s", output_report)
    return output_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CSV datasets before and after cleaning")
    parser.add_argument("--before_csv", required=True)
    parser.add_argument("--after_csv", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_html", default=None)
    args = parser.parse_args()

    out = compare_data(
        before_csv=Path(args.before_csv),
        after_csv=Path(args.after_csv),
        output_report=Path(args.output_report),
        output_html=Path(args.output_html) if args.output_html else None,
    )
    print(out)


if __name__ == "__main__":
    main()
