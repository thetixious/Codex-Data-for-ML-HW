from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def _load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _calc_savings(results: dict[str, list[dict]], metric: str = "f1") -> dict[str, dict[str, float]]:
    if "random" not in results or not results["random"]:
        return {}

    target = results["random"][-1][metric]
    n_random = results["random"][-1]["n_labeled"]

    savings: dict[str, dict[str, float]] = {}
    for strategy, history in results.items():
        if strategy == "random":
            continue
        reached = next((row for row in history if row[metric] >= target), None)
        if reached is None:
            continue
        saved = n_random - reached["n_labeled"]
        savings[strategy] = {
            "target_metric": target,
            "n_random": n_random,
            "n_reached": reached["n_labeled"],
            "saved_examples": saved,
            "saved_pct": 100.0 * saved / max(1, n_random),
        }
    return savings


def generate_reports(history_files: Iterable[Path], labels: Iterable[str], output_img: Path, output_html: Path | None = None, output_conclusion: Path | None = None, metric: str = "f1") -> None:
    history_files = list(history_files)
    labels = list(labels)
    if len(history_files) != len(labels):
        raise ValueError("history_files and labels must have the same length")

    results: dict[str, list[dict]] = {}
    for path, label in zip(history_files, labels):
        results[label] = _load_history(path)

    plt.figure(figsize=(10, 6))
    for label, history in results.items():
        xs = [row["n_labeled"] for row in history]
        ys = [row[metric] for row in history]
        plt.plot(xs, ys, marker="o", label=label)
    plt.title(f"Learning Curves ({metric})")
    plt.xlabel("Labeled samples")
    plt.ylabel(metric.upper())
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img, dpi=140, bbox_inches="tight")
    plt.close()

    savings = _calc_savings(results, metric=metric)

    if output_html is not None:
        all_rows = []
        for label, history in results.items():
            for row in history:
                all_rows.append({"strategy": label, **row})
        hist_df = pd.DataFrame(all_rows)
        fig_metric = px.line(
            hist_df,
            x="n_labeled",
            y=metric,
            color="strategy",
            markers=True,
            title=f"Learning Curves ({metric.upper()})",
            template="plotly_white",
        )
        fig_metric.update_layout(height=430, legend_title_text="Strategy")

        fig_acc = px.line(
            hist_df,
            x="n_labeled",
            y="accuracy",
            color="strategy",
            markers=True,
            title="Learning Curves (ACCURACY)",
            template="plotly_white",
        )
        fig_acc.update_layout(height=430, legend_title_text="Strategy")

        latest = (
            hist_df.sort_values(["strategy", "iteration"])
            .groupby("strategy", as_index=False)
            .tail(1)[["strategy", "n_labeled", "accuracy", "f1"]]
            .sort_values("strategy")
        )
        savings_html = "<p>No savings could be computed against random baseline.</p>"
        if savings:
            savings_df = pd.DataFrame([{"strategy": k, **v} for k, v in savings.items()])
            savings_html = savings_df.to_html(index=False)

        html = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>AL Report</title>
<style>
  :root {{
    --bg:#f5f7fb; --card:#fff; --ink:#1f2937; --muted:#6b7280; --line:#d1d5db; --accent:#0f766e;
  }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Segoe UI,Arial,sans-serif; }}
  .wrap {{ max-width:1200px; margin:0 auto; padding:24px; }}
  .hero {{ background:linear-gradient(120deg,#0f766e,#1d4ed8); color:#fff; border-radius:14px; padding:20px 24px; }}
  .grid {{ display:grid; gap:16px; margin-top:16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }}
  h1,h2 {{ margin:0 0 10px; }}
  p {{ color:var(--muted); margin:0; }}
  table {{ border-collapse:collapse; width:100%; margin-top:8px; }}
  th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; }}
</style>
</head>
<body>
<div class='wrap'>
  <div class='hero'>
    <h1>Active Learning Report</h1>
    <p>Comparison of uncertainty sampling against random baseline.</p>
  </div>
  <div class='grid'>
    <div class='card'>
      <h2>Final Metrics</h2>
      {latest.to_html(index=False)}
    </div>
    <div class='card'>
      {fig_metric.to_html(full_html=False, include_plotlyjs='cdn')}
    </div>
    <div class='card'>
      {fig_acc.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    <div class='card'>
      <h2>Sample Savings vs Random</h2>
      {savings_html}
    </div>
  </div>
</div>
</body></html>"""
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(html, encoding="utf-8")

    if output_conclusion is not None:
        if savings:
            best = max(savings.items(), key=lambda kv: kv[1]["saved_examples"])
            text = (
                "Using Active Learning "
                f"({best[0]}) reached random-baseline quality with "
                f"{best[1]['saved_examples']} fewer labeled examples "
                f"({best[1]['saved_pct']:.1f}%)."
            )
        else:
            text = "Active Learning did not show measurable sample savings against random baseline in this run."
        output_conclusion.parent.mkdir(parents=True, exist_ok=True)
        output_conclusion.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AL learning curve and conclusion")
    parser.add_argument("--history_files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_img", required=True)
    parser.add_argument("--output_html", default=None)
    parser.add_argument("--output_conclusion", default=None)
    parser.add_argument("--metric", default="f1")
    args = parser.parse_args()

    generate_reports(
        history_files=[Path(p) for p in args.history_files],
        labels=args.labels,
        output_img=Path(args.output_img),
        output_html=Path(args.output_html) if args.output_html else None,
        output_conclusion=Path(args.output_conclusion) if args.output_conclusion else None,
        metric=args.metric,
    )
    print(args.output_img)


if __name__ == "__main__":
    main()
