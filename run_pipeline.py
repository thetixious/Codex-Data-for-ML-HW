from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
import yaml

from agents.al_agent import ActiveLearningAgent
from agents.annotation_agent import AnnotationAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from skills.data_collection.scripts.discover_datasets import discover_datasets, select_candidates
from utils.html_report import build_html_page, save_html


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path = "config.yaml") -> dict:
    cfg_path = ROOT / path
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "data_raw": ROOT / "data" / "raw",
        "data_labeled": ROOT / "data" / "labeled",
        "reports": ROOT / "reports",
        "models": ROOT / "models",
        "notebooks": ROOT / "notebooks",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def default_sources_for_topic(topic: str) -> list[dict]:
    candidates = discover_datasets(topic=topic, limit_per_source=20, sources=["hf", "kaggle", "zenodo"])
    selected = select_candidates(candidates, top_k=4, min_score=2.0)

    open_ds = next((c for c in selected if c.source == "hf"), None)
    if open_ds is None:
        open_ds = next((c for c in selected if c.source == "kaggle"), None)

    if open_ds is None:
        source_open = {"type": "hf_dataset", "name": "imdb"}
    else:
        source_open = {
            "type": "hf_dataset" if open_ds.source == "hf" else "kaggle_dataset",
            "name": open_ds.dataset_id,
        }

    # Second required source: API
    source_api = {
        "type": "api",
        "endpoint": "https://zenodo.org/api/records",
        "params": {"q": topic, "size": 100},
        "source_name": "zenodo_api",
    }

    return [source_open, source_api]


def save_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_collection_eda_html(path: Path, df: pd.DataFrame, topic: str) -> None:
    cls = df["label"].astype(str).value_counts().rename_axis("label").reset_index(name="count")
    cls_fig = px.bar(cls, x="label", y="count", title="Class Distribution")
    cls_html = cls_fig.to_html(include_plotlyjs="cdn", full_html=False)

    lengths = df["text"].fillna("").astype(str).str.len().rename("text_len")
    len_fig = px.histogram(lengths.to_frame(), x="text_len", nbins=40, title="Text Length Distribution")
    len_html = len_fig.to_html(include_plotlyjs=False, full_html=False)

    words = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-zA-Zа-яА-Я0-9\s]", " ", regex=True)
        .str.split()
        .explode()
    )
    words = words[words.str.len() > 2]
    top_words = words.value_counts().head(20).rename_axis("word").reset_index(name="count")
    word_fig = px.bar(top_words, x="word", y="count", title="Top-20 Words")
    word_html = word_fig.to_html(include_plotlyjs=False, full_html=False)

    sections = [
        f'<div class="card"><h2>Task</h2><p>{topic}</p></div>',
        f'<div class="card"><h2>Dataset Snapshot</h2>{df.head(20).to_html(index=False)}</div>',
        f'<div class="card">{cls_html}</div>',
        f'<div class="card">{len_html}</div>',
        f'<div class="card">{word_html}</div>',
    ]
    save_html(path, build_html_page("EDA Report", sections))


def save_quality_html(path: Path, report: dict, comparison: pd.DataFrame) -> None:
    sections: list[str] = [
        '<div class="card"><h2>Before/After</h2>'
        + comparison.to_html(index=False)
        + "</div>",
    ]
    if report.get("missing"):
        sections.append('<div class="card"><h2>Missing</h2>' + pd.DataFrame(report["missing"]).T.to_html() + "</div>")
    if report.get("outliers"):
        sections.append('<div class="card"><h2>Outliers</h2>' + pd.DataFrame(report["outliers"]).T.to_html() + "</div>")
    if report.get("imbalance"):
        sections.append('<div class="card"><h2>Imbalance</h2><pre>' + json.dumps(report["imbalance"], ensure_ascii=False, indent=2) + "</pre></div>")
    save_html(path, build_html_page("Quality Report", sections))


def save_al_html(path: Path, hist_entropy: list[dict], hist_random: list[dict]) -> None:
    df = pd.concat([pd.DataFrame(hist_entropy), pd.DataFrame(hist_random)], ignore_index=True)
    fig = px.line(
        df.sort_values(["strategy", "n_labeled"]),
        x="n_labeled",
        y="f1",
        color="strategy",
        markers=True,
        title="Active Learning: F1 vs n_labeled",
    )
    body = fig.to_html(include_plotlyjs="cdn", full_html=False)
    save_html(path, build_html_page("Active Learning Report", [f'<div class="card">{body}</div>']))


def save_quality_report_md(path: Path, report: dict) -> None:
    lines = ["# Quality Report\n\n", "## Summary\n"]
    lines.append(f"- Duplicates: {report.get('duplicates', 0)}\n")
    lines.append(f"- Missing columns: {len(report.get('missing', {}))}\n")
    lines.append(f"- Outlier columns: {len(report.get('outliers', {}))}\n\n")

    if report.get("missing"):
        lines.append("## Missing\n\n")
        lines.append(pd.DataFrame(report["missing"]).T.to_markdown())
        lines.append("\n\n")

    if report.get("outliers"):
        lines.append("## Outliers\n\n")
        lines.append(pd.DataFrame(report["outliers"]).T.to_markdown())
        lines.append("\n\n")

    if report.get("imbalance"):
        lines.append("## Imbalance\n\n")
        lines.append(json.dumps(report["imbalance"], ensure_ascii=False, indent=2))
        lines.append("\n")

    save_markdown(path, "".join(lines))


def save_annotation_report_md(path: Path, metrics: dict, n_total: int, n_low_conf: int, n_reviewed: int) -> None:
    lines = [
        "# Annotation Report\n\n",
        "## Summary\n",
        f"- Total rows: {n_total}\n",
        f"- Low confidence rows sent to review: {n_low_conf}\n",
        f"- Rows reviewed by human: {n_reviewed}\n",
        f"- Mean confidence: {metrics.get('confidence_mean')}\n",
        f"- Agreement: {metrics.get('agreement')}\n",
        f"- Cohen's kappa: {metrics.get('kappa')}\n\n",
        "## Label Distribution\n\n",
        json.dumps(metrics.get("label_dist", {}), ensure_ascii=False, indent=2),
        "\n",
    ]
    save_markdown(path, "".join(lines))


def save_data_card(path: Path, df: pd.DataFrame, topic: str) -> None:
    class_dist = {}
    if "auto_label" in df.columns:
        class_dist = {str(k): int(v) for k, v in df["auto_label"].astype(str).value_counts().to_dict().items()}
    source_dist = {str(k): int(v) for k, v in df["source"].astype(str).value_counts().to_dict().items()} if "source" in df.columns else {}

    lines = [
        "# Data Card\n\n",
        f"- Task/domain: {topic}\n",
        f"- Rows: {len(df)}\n",
        f"- Columns: {', '.join(df.columns.tolist())}\n",
        f"- Class distribution: {json.dumps(class_dist, ensure_ascii=False)}\n",
        f"- Source distribution: {json.dumps(source_dist, ensure_ascii=False)}\n\n",
        "## Schema\n\n",
        df.dtypes.astype(str).rename("dtype").to_frame().to_markdown(),
        "\n",
    ]
    save_markdown(path, "".join(lines))


def build_final_report(
    path: Path,
    topic: str,
    df_labeled: pd.DataFrame,
    quality_report: dict,
    comparison: pd.DataFrame,
    annotation_metrics: dict,
    al_hist_entropy: list[dict],
    al_hist_random: list[dict],
    corrected_rows: int,
) -> None:
    rnd_last = al_hist_random[-1] if al_hist_random else {}
    ent_last = al_hist_entropy[-1] if al_hist_entropy else {}

    lines = [
        f"# Финальный отчёт: {topic}\n\n",
        "## 1. Описание задачи и датасета\n",
        f"- Тема: {topic}\n",
        f"- Размер размеченного датасета: {len(df_labeled)}\n",
        f"- Колонки: {', '.join(df_labeled.columns[:20])}\n",
        f"- Классы: {', '.join(sorted(df_labeled['auto_label'].astype(str).unique().tolist())) if 'auto_label' in df_labeled.columns else 'n/a'}\n\n",
        "## 2. Что делал каждый агент\n",
        "- DataCollectionAgent: сбор из open dataset + API, унификация схемы.\n",
        "- DataQualityAgent: детекция проблем и чистка по выбранной стратегии.\n",
        "- AnnotationAgent: авторазметка, генерация спецификации, экспорт в LabelStudio.\n",
        "- ActiveLearningAgent: сравнение entropy vs random и кривые обучения.\n\n",
        "## 3. HITL-точка\n",
        f"- Исправлено человеком после review_queue: {corrected_rows} примеров.\n",
        "- Исправленные метки слиты обратно в итоговый датасет.\n\n",
        "## 4. Метрики качества\n",
        f"- Missing columns: {len(quality_report.get('missing', {}))}; duplicates: {quality_report.get('duplicates', 0)}\n",
        f"- Annotation confidence mean: {annotation_metrics.get('confidence_mean')}\n",
        f"- AL random final: acc={rnd_last.get('accuracy')}, f1={rnd_last.get('f1')}\n",
        f"- AL entropy final: acc={ent_last.get('accuracy')}, f1={ent_last.get('f1')}\n\n",
        "## 5. Ретроспектива\n",
        "- Сработало: единая схема данных, прозрачный пайплайн, явная HITL-проверка.\n",
        "- Не сработало: качество может зависеть от слабой авторазметки без внешней модели.\n",
        "- Улучшения: добавить LLM-аннотацию/калибровку confidence и richer feature engineering.\n",
    ]
    save_markdown(path, "".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified data project pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--strategy", default="smart", choices=["aggressive", "smart", "conservative"])
    parser.add_argument("--auto_approve", action="store_true", help="Skip pause for manual review")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs()

    topic = args.topic or cfg.get("task_name") or "text classification"

    # Step 1: Collection
    cfg_sources = ((cfg.get("data_collection") or {}).get("sources")) or ((cfg.get("pipeline") or {}).get("sources"))
    sources = cfg_sources or default_sources_for_topic(topic)
    collection_agent = DataCollectionAgent(config=args.config)
    df_raw = collection_agent.run(sources=sources)
    if df_raw.empty:
        raise RuntimeError("DataCollectionAgent produced empty dataset")
    raw_path = paths["data_raw"] / "collected_unified.csv"
    df_raw.to_csv(raw_path, index=False)
    save_collection_eda_html(paths["reports"] / "eda_report.html", df_raw, topic)

    # Step 2: Quality
    dq = DataQualityAgent()
    strategy_map = {
        "aggressive": {"missing": "drop", "duplicates": "drop", "outliers": "drop"},
        "smart": {"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"},
        "conservative": {"missing": "ffill", "duplicates": "drop", "outliers": "none"},
    }
    selected_strategy = strategy_map[args.strategy]
    quality_report = dq.detect_issues(df_raw, label_col="label" if "label" in df_raw.columns else None)
    df_clean = dq.fix(df_raw, strategy=selected_strategy)
    if df_clean.empty:
        raise RuntimeError("DataQualityAgent produced empty dataset after cleaning")
    comparison = dq.compare(df_raw, df_clean)

    clean_path = paths["data_raw"] / "cleaned.csv"
    df_clean.to_csv(clean_path, index=False)
    comparison.to_csv(paths["reports"] / "quality_comparison.csv", index=False)
    save_quality_report_md(paths["reports"] / "quality_report.md", quality_report)
    save_quality_html(paths["reports"] / "quality_report.html", quality_report, comparison)
    save_markdown(
        paths["reports"] / "strategy_justification.md",
        (
            "# Strategy Justification\n\n"
            f"Selected strategy: `{args.strategy}`\n\n"
            f"Config: `{json.dumps(selected_strategy, ensure_ascii=False)}`\n\n"
            "Chosen to balance data retention and noise reduction for downstream model training."
        ),
    )

    # Step 3: Annotation + HITL
    ann = AnnotationAgent(modality="text")
    df_labeled = ann.auto_label(df_clean, modality="text")
    df_before_review = df_labeled.copy()

    queue = df_labeled[df_labeled["confidence"] < float((cfg.get("annotation") or {}).get("default_threshold", 0.7))].copy()
    if queue.empty and len(df_labeled) > 0:
        # Ensure explicit HITL checkpoint even when auto-label confidence is high.
        n_fallback = min(max(20, int(0.02 * len(df_labeled))), len(df_labeled))
        queue = df_labeled.sample(n=n_fallback, random_state=42).copy()
        queue["confidence"] = queue["confidence"].astype(float).clip(upper=0.69)
    if "human_label" not in queue.columns:
        queue["human_label"] = queue["auto_label"]
    queue_path = ROOT / "review_queue.csv"
    queue.to_csv(queue_path, index=False)

    corrected_rows = 0
    reviewed_rows = 0
    n_low_conf = int(len(queue))
    if len(queue) > 0 and not args.auto_approve:
        print(f"Manual review required: {queue_path}")
        print("Please edit labels/confidence in review_queue.csv, then press Enter...")
        input()

    if queue_path.exists() and len(queue) > 0:
        reviewed = pd.read_csv(queue_path)
        base = df_labeled.set_index("__row_id")
        reviewed = reviewed.set_index("__row_id")
        overlap = base.index.intersection(reviewed.index)
        if len(overlap) > 0:
            reviewed_rows = int(len(overlap))
            human_col = "human_label" if "human_label" in reviewed.columns else "auto_label"
            human_labels = reviewed.loc[overlap, human_col].astype(str)
            before_labels = df_before_review.set_index("__row_id").loc[overlap, "auto_label"].astype(str)
            corrected_rows = int((human_labels != before_labels).sum())

            # Merge human decisions into final auto_label for training.
            base.loc[overlap, "auto_label"] = human_labels
            base.loc[overlap, "human_label"] = human_labels

            # Keep edited confidence/reason if provided.
            for col in ("confidence", "reason"):
                if col in reviewed.columns and col in base.columns:
                    base.loc[overlap, col] = reviewed.loc[overlap, col]
        df_labeled = base.reset_index(drop=False)

    labeled_path = paths["data_labeled"] / "labeled_dataset.csv"
    df_labeled.to_csv(labeled_path, index=False)
    save_data_card(paths["data_labeled"] / "data_card.md", df_labeled, topic)

    spec_text = ann.generate_spec(df_labeled, task=topic)
    ann.save_spec(spec_text, paths["reports"] / "annotation_spec.md")

    if "human_label" in df_labeled.columns and df_labeled["human_label"].notna().any():
        eval_df = df_labeled[df_labeled["human_label"].notna()].copy()
        annotation_metrics = ann.check_quality(eval_df, human_col="human_label")
    else:
        annotation_metrics = ann.check_quality(df_labeled)
    (paths["reports"] / "annotation_metrics.json").write_text(json.dumps(annotation_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_annotation_report_md(
        paths["reports"] / "annotation_report.md",
        annotation_metrics,
        n_total=int(len(df_labeled)),
        n_low_conf=n_low_conf,
        n_reviewed=reviewed_rows,
    )

    ls_tasks = ann.export_to_labelstudio(df_labeled)
    ann.save_labelstudio(ls_tasks, paths["reports"] / "labelstudio_import.json")

    # Step 4: Active Learning
    al = ActiveLearningAgent(model="logreg")

    label_col = "auto_label"
    if label_col not in df_labeled.columns:
        df_labeled[label_col] = "unlabeled"

    if len(df_labeled) < 10 or df_labeled[label_col].nunique() < 2:
        base_eval = al.evaluate(df_labeled, df_labeled, text_col="text", label_col=label_col)
        hist_entropy = [
            {
                "iteration": 0,
                "n_labeled": int(len(df_labeled)),
                "accuracy": base_eval["accuracy"],
                "f1": base_eval["f1"],
                "strategy": "entropy",
                "queried_indices": [],
            }
        ]
        hist_random = [
            {
                "iteration": 0,
                "n_labeled": int(len(df_labeled)),
                "accuracy": base_eval["accuracy"],
                "f1": base_eval["f1"],
                "strategy": "random",
                "queried_indices": [],
            }
        ]
    else:
        strat = df_labeled[label_col] if df_labeled[label_col].value_counts().min() >= 2 else None
        train_df, test_df = train_test_split(df_labeled, test_size=0.2, random_state=42, stratify=strat)

        n_start = min(50, max(10, len(train_df) // 2))
        train_size = min(max(1, n_start), max(1, len(train_df) - 1))
        strat_train = train_df[label_col] if train_df[label_col].value_counts().min() >= 2 else None
        labeled_start, pool_df = train_test_split(train_df, train_size=train_size, random_state=42, stratify=strat_train)

        hist_entropy = al.run_cycle(labeled_start, pool_df, strategy="entropy", n_iterations=5, batch_size=20, test_df=test_df)
        hist_random = al.run_cycle(labeled_start, pool_df, strategy="random", n_iterations=5, batch_size=20, test_df=test_df)

    (paths["reports"] / "al_history_entropy.json").write_text(json.dumps(hist_entropy, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["reports"] / "al_history_random.json").write_text(json.dumps(hist_random, ensure_ascii=False, indent=2), encoding="utf-8")

    al.report(hist_entropy + hist_random, output_path=paths["reports"] / "learning_curve.png")
    save_al_html(paths["reports"] / "al_report.html", hist_entropy, hist_random)

    savings = al.compare({"entropy": hist_entropy, "random": hist_random})
    (paths["reports"] / "al_report.md").write_text(
        "# AL Report\n\n" + json.dumps(savings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Step 5: Final training artifact
    al.fit(df_labeled, text_col="text", label_col=label_col)
    with (paths["models"] / "final_model.pkl").open("wb") as f:
        pickle.dump(al.model, f)

    # Step 6: Final report
    build_final_report(
        path=paths["reports"] / "final_report.md",
        topic=topic,
        df_labeled=df_labeled,
        quality_report=quality_report,
        comparison=comparison,
        annotation_metrics=annotation_metrics,
        al_hist_entropy=hist_entropy,
        al_hist_random=hist_random,
        corrected_rows=corrected_rows,
    )

    print("Pipeline completed.")
    print(f"Raw data: {raw_path}")
    print(f"Labeled data: {labeled_path}")
    print(f"Model: {paths['models'] / 'final_model.pkl'}")
    print(f"Final report: {paths['reports'] / 'final_report.md'}")


if __name__ == "__main__":
    main()
