from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd
import yaml


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_dataset_info(run_root: Path) -> dict[str, str]:
    out = {
        "rows": "n/a",
        "cols": "n/a",
        "target_classes": "n/a",
    }
    labeled = run_root / "data" / "labeled_dataset.csv"
    if labeled.exists():
        df = pd.read_csv(labeled)
        out["rows"] = str(len(df))
        out["cols"] = str(df.shape[1])
        if "auto_label" in df.columns:
            out["target_classes"] = ", ".join(sorted(df["auto_label"].astype(str).dropna().unique().tolist())[:20])
    return out


def build_report(run_root: Path, output_readme: Path) -> Path:
    run_cfg_path = run_root / "run_config.yaml"
    run_cfg = yaml.safe_load(run_cfg_path.read_text(encoding="utf-8")) if run_cfg_path.exists() else {}

    ds_info = _detect_dataset_info(run_root)
    quality = _read_json(run_root / "reports" / "quality_report.json") or {}
    al_random = _read_json(run_root / "reports" / "al_history_random.json") or []
    al_entropy = _read_json(run_root / "reports" / "al_history_entropy.json") or []
    ann_metrics = _read_json(run_root / "reports" / "annotation_metrics.json") or {}

    al_random_last = al_random[-1] if al_random else {}
    al_entropy_last = al_entropy[-1] if al_entropy else {}

    hitl_rows = "n/a"
    review_queue = run_root / "data" / "review_queue.csv"
    if review_queue.exists():
        hitl_rows = str(max(0, sum(1 for _ in review_queue.open("r", encoding="utf-8")) - 1))

    conclusion_txt = ""
    cpath = run_root / "reports" / "al_conclusion.txt"
    if cpath.exists():
        conclusion_txt = cpath.read_text(encoding="utf-8").strip()

    lines = [
        f"# Final Pipeline Report: {run_cfg.get('task_name', run_root.name)}\n\n",
        "## 1. Task and Dataset\n",
        f"- Task: {run_cfg.get('task_description', 'n/a')}\n",
        f"- Labeled rows: {ds_info['rows']}\n",
        f"- Columns: {ds_info['cols']}\n",
        f"- Classes/labels: {ds_info['target_classes']}\n\n",
        "## 2. Stage Decisions\n",
        "- Data collection: multi-source ingestion + schema unification completed.\n",
        f"- Data quality: issue types detected = {quality.get('issue_types_detected', 'n/a')}, strategy selected by user at checkpoint.\n",
        "- Annotation: auto-labeling done using explicit rules and confidence scoring.\n",
        "- Active learning: entropy vs random comparison completed.\n\n",
        "## 3. Human-in-the-Loop Checkpoint\n",
        f"- Rows sent to manual review (confidence threshold): {hitl_rows}\n",
        "- Reviewed queue merged back into final labeled dataset.\n\n",
        "## 4. Metrics by Stage\n",
        f"- Quality: duplicates = {quality.get('duplicates', 'n/a')}, missing columns with gaps = {len((quality.get('missing') or {}))}.\n",
        f"- Annotation: kappa = {ann_metrics.get('kappa', 'n/a')}, mean confidence = {ann_metrics.get('confidence_mean', 'n/a')}.\n",
        f"- AL random final: accuracy = {al_random_last.get('accuracy', 'n/a')}, f1 = {al_random_last.get('f1', 'n/a')}.\n",
        f"- AL entropy final: accuracy = {al_entropy_last.get('accuracy', 'n/a')}, f1 = {al_entropy_last.get('f1', 'n/a')}.\n",
    ]

    if conclusion_txt:
        lines.append(f"- AL conclusion: {conclusion_txt}\n")

    lines.extend(
        [
            "\n## 5. Retrospective\n",
            "- What worked: isolated runs, stage checkpoints, reproducible artifacts, and AL comparison history.\n",
            "- What did not: performance may vary by label quality and class imbalance in initial labeled subset.\n",
            "- What to improve next: stronger base model, richer annotation policy, and larger validated seed set.\n",
        ]
    )

    output_readme.parent.mkdir(parents=True, exist_ok=True)
    output_readme.write_text("".join(lines), encoding="utf-8")
    return output_readme


def build_report_html(run_root: Path, output_html: Path) -> Path:
    run_cfg_path = run_root / "run_config.yaml"
    run_cfg = yaml.safe_load(run_cfg_path.read_text(encoding="utf-8")) if run_cfg_path.exists() else {}

    ds_info = _detect_dataset_info(run_root)
    quality = _read_json(run_root / "reports" / "quality_report.json") or {}
    al_random = _read_json(run_root / "reports" / "al_history_random.json") or []
    al_entropy = _read_json(run_root / "reports" / "al_history_entropy.json") or []
    ann_metrics = _read_json(run_root / "reports" / "annotation_metrics.json") or {}

    al_random_last = al_random[-1] if al_random else {}
    al_entropy_last = al_entropy[-1] if al_entropy else {}

    hitl_rows = "n/a"
    review_queue = run_root / "data" / "review_queue.csv"
    if review_queue.exists():
        hitl_rows = str(max(0, sum(1 for _ in review_queue.open("r", encoding="utf-8")) - 1))

    links = [
        ("EDA HTML", "reports/eda_report.html"),
        ("Quality HTML", "reports/quality_report.html"),
        ("AL HTML", "reports/al_learning_curves.html"),
        ("Annotation Spec", "labeling/annotation_spec.md"),
        ("LabelStudio JSON", "labeling/labelstudio_import.json"),
        ("Labeled Dataset", "data/labeled_dataset.csv"),
    ]
    links_html = "".join(
        f"<li><a href='{rel}'>{title}</a></li>" for title, rel in links if (run_root / rel).exists()
    )

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Final Pipeline Report</title>
  <style>
    :root {{
      --bg:#f4f6fb; --card:#fff; --ink:#1f2937; --muted:#6b7280; --line:#d1d5db; --accent:#0f766e;
    }}
    body {{ margin:0; font-family:Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--bg); }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:24px; }}
    .hero {{ border-radius:16px; padding:24px; color:#fff; background:linear-gradient(120deg,#0f766e,#1d4ed8); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; margin-top:14px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }}
    h1,h2,h3 {{ margin:0 0 8px; }}
    p {{ margin:0; color:var(--muted); }}
    ul {{ margin:8px 0 0 18px; }}
    li {{ margin:3px 0; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border:1px solid var(--line); padding:6px 8px; text-align:left; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='hero'>
      <h1>Final Pipeline Report</h1>
      <p>{run_cfg.get('task_name', run_root.name)}</p>
    </div>
    <div class='grid'>
      <div class='card'>
        <h3>Dataset</h3>
        <table>
          <tr><th>Rows</th><td>{ds_info['rows']}</td></tr>
          <tr><th>Columns</th><td>{ds_info['cols']}</td></tr>
          <tr><th>Classes</th><td>{ds_info['target_classes']}</td></tr>
        </table>
      </div>
      <div class='card'>
        <h3>Quality</h3>
        <table>
          <tr><th>Issue Types</th><td>{quality.get('issue_types_detected', 'n/a')}</td></tr>
          <tr><th>Duplicates</th><td>{quality.get('duplicates', 'n/a')}</td></tr>
          <tr><th>Missing Columns</th><td>{len((quality.get('missing') or {}))}</td></tr>
        </table>
      </div>
      <div class='card'>
        <h3>Annotation + HITL</h3>
        <table>
          <tr><th>Queue Rows</th><td>{hitl_rows}</td></tr>
          <tr><th>Kappa</th><td>{ann_metrics.get('kappa', 'n/a')}</td></tr>
          <tr><th>Mean Confidence</th><td>{ann_metrics.get('confidence_mean', 'n/a')}</td></tr>
        </table>
      </div>
      <div class='card'>
        <h3>Active Learning</h3>
        <table>
          <tr><th>Random F1</th><td>{al_random_last.get('f1', 'n/a')}</td></tr>
          <tr><th>Entropy F1</th><td>{al_entropy_last.get('f1', 'n/a')}</td></tr>
          <tr><th>Random Accuracy</th><td>{al_random_last.get('accuracy', 'n/a')}</td></tr>
          <tr><th>Entropy Accuracy</th><td>{al_entropy_last.get('accuracy', 'n/a')}</td></tr>
        </table>
      </div>
      <div class='card' style='grid-column:1/-1'>
        <h3>Artifacts</h3>
        <ul>{links_html}</ul>
      </div>
    </div>
  </div>
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final run README with 5 required sections")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_readme", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_readme = Path(args.output_readme) if args.output_readme else run_root / "README.md"
    out = build_report(run_root, output_readme)
    build_report_html(run_root, run_root / "reports" / "final_report.html")
    print(out)


if __name__ == "__main__":
    main()
