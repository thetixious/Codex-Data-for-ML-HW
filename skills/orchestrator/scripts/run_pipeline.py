from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Callable

import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from skills.active_learning.scripts.run_experiment import run_experiment
from skills.annotation.scripts.auto_label import auto_label
from skills.annotation.scripts.build_review_queue import build_review_queue
from skills.annotation.scripts.check_quality import calculate_metrics
from skills.annotation.scripts.export_to_labelstudio import export_labelstudio
from skills.annotation.scripts.generate_spec import generate_spec
from skills.annotation.scripts.merge_reviewed import merge_reviewed
from skills.data_collection.scripts.discover_datasets import discover_datasets, select_candidates
from skills.data_collection.scripts.download_hf import download_hf_dataset
from skills.data_collection.scripts.download_kaggle import download_kaggle_dataset
from skills.data_collection.scripts.download_web import download_web_file
from skills.data_collection.scripts.generate_eda_report import generate_eda_report
from skills.data_collection.scripts.merge_datasets import merge_csv_files
from skills.data_collection.scripts.unify_and_process import unify_dataframe
from skills.data_quality.scripts.compare_datasets import compare_data
from skills.data_quality.scripts.detect_issues import detect_issues
from skills.data_quality.scripts.fix_data import fix_data
from skills.data_quality.scripts.save_strategy_justification import save_justification
from skills.orchestrator.scripts.build_final_report import build_report, build_report_html
from utils.run_context import RunPaths, make_run_folder


STAGES = ["data_collection", "data_quality", "annotation", "active_learning", "final_report"]


TARGET_HINTS = {"label", "target", "class", "outcome", "status", "approved", "denied", "default"}
LEAK_PRONE_COL_HINTS = {
    "education",
    "gender",
    "married",
    "dependents",
    "loan_id",
    "id",
    "uuid",
    "name",
}
TEXT_HINTS = {"text", "comment", "review", "content", "body", "message", "title", "question", "prompt"}
VALUE_HINTS = {"price", "amount", "cost", "value", "score", "rating"}
YEAR_HINTS = {"year", "date", "timestamp", "time"}


def _load_cfg() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _ensure_run_paths(run_root: Path) -> RunPaths:
    paths = RunPaths(
        root=run_root,
        data=run_root / "data",
        reports=run_root / "reports",
        models=run_root / "models",
        notebooks=run_root / "notebooks",
        labeling=run_root / "labeling",
        scripts=run_root / "scripts",
    )
    for p in [paths.root, paths.data, paths.reports, paths.models, paths.notebooks, paths.labeling, paths.scripts]:
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{prompt} {suffix}: ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes", "да", "д"}


def _checkpoint(title: str, auto_confirm: bool) -> None:
    if auto_confirm:
        return
    ok = _ask_yes_no(f"Checkpoint '{title}' completed. Continue to next stage?", default=False)
    if not ok:
        raise SystemExit("Stopped by user at checkpoint.")


def _select_strategy(cfg: dict, args: argparse.Namespace) -> str:
    if args.quality_strategy:
        return args.quality_strategy
    if args.auto_confirm:
        return (cfg.get("quality") or {}).get("default_strategy", "smart")

    choices = ["aggressive", "smart", "conservative"]
    print("Available cleaning strategies:")
    for i, s in enumerate(choices, 1):
        print(f"  [{i}] {s}")
    raw = input("Choose strategy [1-3, default=2]: ").strip()
    if raw in {"1", "2", "3"}:
        return choices[int(raw) - 1]
    return "smart"


def _norm(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")


def _task_tokens(task: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", task.lower()))


def _infer_rename_keep(df: pd.DataFrame, task_description: str, max_cols: int = 30) -> tuple[dict[str, str], list[str]]:
    tokens = _task_tokens(task_description)
    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()

    def _try_map(col: str, target: str) -> None:
        if target in used_targets:
            return
        rename_map[col] = target
        used_targets.add(target)

    for col in df.columns:
        c = _norm(col)
        parts = set(c.split("_"))

        if parts & TARGET_HINTS:
            _try_map(col, "label")
            continue
        if parts & TEXT_HINTS:
            _try_map(col, "text")
            continue
        if parts & VALUE_HINTS:
            _try_map(col, "value")
            continue
        if parts & YEAR_HINTS:
            _try_map(col, "year")
            continue

    ranked: list[tuple[float, str]] = []
    for col in df.columns:
        c = _norm(col)
        parts = set(c.split("_"))
        score = 0.0
        if col in rename_map:
            score += 3.0
        if tokens & parts:
            score += 2.0
        if pd.api.types.is_numeric_dtype(df[col]):
            score += 1.0
        else:
            score += 0.7
        missing_ratio = float(df[col].isna().mean())
        score += max(0.0, 1.0 - missing_ratio)
        ranked.append((score, col))

    ranked.sort(reverse=True)
    keep_cols = [col for _, col in ranked[:max_cols]]

    return rename_map, keep_cols


def _download_source(source: dict, out_dir: Path) -> Path:
    stype = source.get("type")
    sid = source.get("name") or source.get("url")
    if stype not in {"api"} and not sid:
        raise ValueError("Source has no name/url")

    if stype == "hf":
        return download_hf_dataset(
            dataset_name=sid,
            output_dir=out_dir,
            subset=source.get("subset"),
            split=source.get("split"),
        )
    if stype == "kaggle":
        return download_kaggle_dataset(dataset_name=sid, output_dir=out_dir)
    if stype == "api":
        endpoint = source.get("endpoint")
        if not endpoint:
            raise ValueError("API source requires endpoint")
        params = source.get("params") or {}
        return _download_api_source(endpoint=endpoint, params=params, output_dir=out_dir, source_name=source.get("source_name", "api"))
    if stype == "web":
        return download_web_file(url=sid, output_dir=out_dir)

    raise ValueError(f"Unsupported source type: {stype}")


def _download_api_source(endpoint: str, params: dict, output_dir: Path, source_name: str) -> Path:
    attempts: list[dict] = [dict(params)]
    if "size" in params:
        p = dict(params)
        p["size"] = 10
        attempts.append(p)
    attempts.append({k: v for k, v in params.items() if k not in {"size", "type"}})
    attempts.append({})

    payload = None
    for p in attempts:
        try:
            resp = requests.get(endpoint, params=p, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception:
            continue
    if payload is None:
        raise RuntimeError(f"Failed to fetch API source: {endpoint}")

    rows: list[dict] = []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("data", "results", "items", "records"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if not rows and isinstance(payload.get("hits"), dict) and isinstance(payload["hits"].get("hits"), list):
            rows = payload["hits"]["hits"]
        if not rows:
            rows = [payload]

    df = pd.json_normalize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", source_name).strip("_")
    out_path = output_dir / f"api_{safe_name}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _discover_sources(cfg: dict, topic: str, reports_dir: Path) -> list[dict]:
    dcfg = cfg.get("data_collection") or {}
    discover_cfg = dcfg.get("discovery") or {}

    src_list = discover_cfg.get("sources") or ["hf", "kaggle", "zenodo"]
    limit = int(discover_cfg.get("limit_per_source", 20))
    top_k = int(discover_cfg.get("select_top_k", 4))
    min_score = float(discover_cfg.get("min_score", 2.0))

    candidates = discover_datasets(topic=topic, limit_per_source=limit, sources=src_list)
    selected = select_candidates(candidates, top_k=top_k, min_score=min_score)

    payload = {
        "topic": topic,
        "sources": src_list,
        "candidates": [c.__dict__ for c in candidates],
        "selected": [c.__dict__ for c in selected],
    }
    (reports_dir / "discovery_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Convert selected candidates into executable sources.
    exec_sources: list[dict] = []
    has_open_dataset = False
    added_zenodo_api = False
    for item in selected:
        if item.source in {"hf", "kaggle"}:
            has_open_dataset = True
            exec_sources.append(
                {
                    "type": item.source,
                    "name": item.dataset_id,
                    "source_name": f"{item.source}:{item.name}",
                    "rename_map": {},
                    "keep_cols": [],
                }
            )
        elif item.source == "zenodo" and not added_zenodo_api:
            added_zenodo_api = True
            exec_sources.append(
                {
                    "type": "api",
                    "name": f"zenodo:{item.dataset_id}",
                    "endpoint": "https://zenodo.org/api/records",
                    "params": {"q": topic, "size": 50},
                    "source_name": "api:zenodo",
                    "rename_map": {},
                    "keep_cols": [],
                }
            )

    if not has_open_dataset:
        # Reproducible fallback open dataset.
        exec_sources.append(
            {
                "type": "hf",
                "name": "13nishit/LoanApprovalPrediction",
                "source_name": "hf:LoanApprovalPrediction",
                "rename_map": {},
                "keep_cols": [],
            }
        )

    if not any(s.get("type") in {"api", "web"} for s in exec_sources):
        exec_sources.append(
            {
                "type": "api",
                "name": "zenodo_topic_api",
                "endpoint": "https://zenodo.org/api/records",
                "params": {"q": topic, "size": 50},
                "source_name": "api:zenodo",
                "rename_map": {},
                "keep_cols": [],
            }
        )

    return exec_sources


def _default_annotation_rules(df: pd.DataFrame, column: str) -> list[dict]:
    if column not in df.columns:
        raise KeyError(f"Annotation column '{column}' not found")
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(s) < 20:
        raise ValueError("Not enough numeric values to auto-generate annotation rules")

    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75]).tolist()
    return [
        {"type": "threshold", "op": "<", "val": float(q1), "label": "tier_1", "conf": 0.9},
        {"type": "range", "min": float(q1), "max": float(q2), "label": "tier_2", "conf": 0.85},
        {"type": "range", "min": float(q2), "max": float(q3), "label": "tier_3", "conf": 0.75},
        {"type": "threshold", "op": ">=", "val": float(q3), "label": "tier_4", "conf": 0.65},
    ]


def _detect_label_column(df: pd.DataFrame) -> str | None:
    preferred = ["label", "target", "class", "loan_status", "status", "outcome", "default_flag", "approved"]
    lower_map = {c.lower(): c for c in df.columns}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]

    for col in df.columns:
        c = _norm(col)
        parts = set(c.split("_"))
        if parts & LEAK_PRONE_COL_HINTS:
            continue
        if parts & TARGET_HINTS:
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique <= 1000:
                return col

    candidates = []
    for col in df.columns:
        c = _norm(col)
        if set(c.split("_")) & LEAK_PRONE_COL_HINTS:
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 50:
            candidates.append((nunique, col))
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None


def _auto_label_from_existing_col(df: pd.DataFrame, label_col: str, out_path: Path) -> Path:
    out = df.copy().reset_index(drop=True)
    if "__row_id" not in out.columns:
        out["__row_id"] = out.index.astype(int)
    out["auto_label"] = out[label_col].astype(str)
    out["confidence"] = 0.65
    out["reason"] = f"copied_from:{label_col}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def _choose_annotation_column(df: pd.DataFrame, cfg: dict) -> str | None:
    acfg = cfg.get("annotation") or {}
    preferred = acfg.get("column")
    if preferred and preferred in df.columns:
        return preferred

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 20]
    for col in numeric_cols:
        parts = set(_norm(col).split("_"))
        if parts & VALUE_HINTS:
            return col
    if numeric_cols:
        return numeric_cols[0]
    return None


def _stage_data_collection(cfg: dict, paths: RunPaths, args: argparse.Namespace) -> None:
    dcfg = cfg.get("data_collection") or {}
    min_sources = int((cfg.get("run") or {}).get("min_sources", 2))
    max_cols = int((dcfg.get("unify") or {}).get("max_cols", 30))

    topic = args.topic or (dcfg.get("discovery") or {}).get("topic") or cfg.get("task_description") or cfg.get("task_name") or "dataset search"

    manual_sources = dcfg.get("sources") or []
    if manual_sources:
        sources = manual_sources
    else:
        sources = _discover_sources(cfg, topic=topic, reports_dir=paths.reports)

    has_open_dataset = any((s.get("type") in {"hf", "kaggle"}) for s in sources)
    if not has_open_dataset:
        raise ValueError("Data collection requires at least one open dataset source (hf/kaggle).")
    if not any((s.get("type") in {"api", "web"}) for s in sources):
        sources.append(
            {
                "type": "api",
                "name": "zenodo_topic_api",
                "endpoint": "https://zenodo.org/api/records",
                "params": {"q": topic, "size": 50},
                "source_name": "api:zenodo",
                "rename_map": {},
                "keep_cols": [],
            }
        )

    if len(sources) < min_sources:
        raise ValueError(
            f"Not enough downloadable sources found ({len(sources)}). "
            "Provide manual data_collection.sources in config or broader topic."
        )

    run_timestamp = paths.root.name.split("_", 1)[-1] if "_" in paths.root.name else paths.root.name
    keep_cols_default = dcfg.get("default_keep_cols") or []

    manifest: list[dict] = []
    successful = 0
    successful_types: list[str] = []

    for i, source in enumerate(sources, 1):
        try:
            raw_path = _download_source(source, paths.data)
        except Exception as exc:
            manifest.append({"source_idx": i, "source": source, "status": "download_failed", "error": str(exc)})
            continue

        raw_df = pd.read_csv(raw_path)

        rename_map = source.get("rename_map") or {}
        keep_cols = source.get("keep_cols") or keep_cols_default
        if not rename_map or not keep_cols:
            auto_rename, auto_keep = _infer_rename_keep(raw_df, task_description=topic, max_cols=max_cols)
            rename_map = rename_map or auto_rename
            keep_cols = keep_cols or auto_keep

        unified_path = paths.data / f"unified_{i}.csv"
        unify_dataframe(
            input_path=raw_path,
            output_path=unified_path,
            rename_map=rename_map,
            keep_cols=keep_cols,
            source_name=source.get("source_name") or f"source_{i}",
            run_timestamp=run_timestamp,
            drop_duplicates=True,
        )

        successful += 1
        successful_types.append(str(source.get("type")))
        manifest.append(
            {
                "source_idx": i,
                "source": source,
                "status": "ok",
                "raw_path": str(raw_path),
                "unified_path": str(unified_path),
                "rename_map": rename_map,
                "keep_cols": keep_cols,
            }
        )

    if successful < min_sources:
        raise RuntimeError(f"Only {successful} sources processed successfully, need at least {min_sources}")
    if "api" not in successful_types and "web" not in successful_types:
        raise RuntimeError("Need at least one successful API/web source for assignment requirements.")

    merged_path = paths.data / "merged_dataset.csv"
    merge_csv_files(paths.data, merged_path, pattern="unified_*.csv")

    generate_eda_report(merged_path, paths.reports / "eda_report.html", topic)
    (paths.reports / "data_collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _stage_data_quality(cfg: dict, paths: RunPaths, args: argparse.Namespace) -> None:
    merged = paths.data / "merged_dataset.csv"
    if not merged.exists():
        raise FileNotFoundError(f"Missing input for quality stage: {merged}")

    target_col = ((cfg.get("schema") or {}).get("target_col") or "auto_label")
    detect_issues(merged, paths.reports, label_col=target_col)

    strategy = _select_strategy(cfg, args)
    cleaned = paths.data / "cleaned_dataset.csv"
    fix_data(merged, cleaned, strategy=strategy)

    compare_data(
        before_csv=merged,
        after_csv=cleaned,
        output_report=paths.reports / "quality_comparison.md",
        output_html=paths.reports / "quality_comparison.html",
    )

    rationale = args.strategy_rationale
    if not rationale:
        if args.auto_confirm:
            rationale = (cfg.get("quality") or {}).get("default_rationale", "Chosen as default balanced strategy.")
        else:
            rationale = input("Provide short rationale for selected strategy: ").strip() or "Chosen after reviewing quality report."
    save_justification(strategy=strategy, rationale=rationale, output_path=paths.reports / "strategy_justification.md")


def _stage_annotation(cfg: dict, paths: RunPaths, args: argparse.Namespace) -> None:
    cleaned = paths.data / "cleaned_dataset.csv"

    acfg = cfg.get("annotation") or {}
    threshold = float(acfg.get("default_threshold", 0.7))
    auto_path = paths.data / "auto_labeled_dataset.csv"
    mode = args.annotation_mode
    lineage_path = paths.reports / "annotation_lineage.json"
    label_source_col: str | None = None
    label_source_method = "rules"
    allow_copy_from_existing = bool(acfg.get("allow_copy_from_existing_label", False))

    if mode in {"full", "queue_only"}:
        if not cleaned.exists():
            raise FileNotFoundError(f"Missing input for annotation stage: {cleaned}")

        clean_df = pd.read_csv(cleaned)
        rules = acfg.get("rules")
        column = _choose_annotation_column(clean_df, cfg)

        label_col = _detect_label_column(clean_df)
        if allow_copy_from_existing and not rules and label_col and label_col != "auto_label":
            _auto_label_from_existing_col(clean_df, label_col=label_col, out_path=auto_path)
            label_source_col = label_col
            label_source_method = "copied_existing_label"
        else:
            if column is None:
                raise ValueError("Could not determine annotation column; specify annotation.column in config")
            if not rules:
                rules = _default_annotation_rules(clean_df, column)
            auto_label(
                input_csv=cleaned,
                output_csv=auto_path,
                column=column,
                rules=rules,
                fallback_label=acfg.get("fallback_label", "Unknown"),
                fallback_conf=float(acfg.get("fallback_conf", 0.5)),
            )
            label_source_col = column
            label_source_method = "rule_based"

        queue_path = paths.data / "review_queue.csv"
        build_review_queue(auto_path, queue_path, threshold=threshold, confidence_col="confidence")

        queue_df = pd.read_csv(queue_path)
        if queue_df.empty:
            # Ensure real HITL checkpoint even when confidence is high.
            base = pd.read_csv(auto_path)
            n_force = min(max(20, int(0.02 * len(base))), len(base))
            if n_force > 0:
                queue_df = base.sample(n=n_force, random_state=42).copy()
                queue_df["confidence"] = pd.to_numeric(queue_df["confidence"], errors="coerce").fillna(0.69).clip(upper=0.69)
                queue_df.to_csv(queue_path, index=False)

        queue_rows = max(0, sum(1 for _ in queue_path.open("r", encoding="utf-8")) - 1)
        print(f"HITL queue prepared: {queue_path} (rows={queue_rows})")

        if mode == "queue_only":
            print("Annotation stage stopped after queue generation (annotation_mode=queue_only).")
            return

        if queue_rows > 0 and not args.no_review_pause:
            print("\nManual review required.")
            print(f"1) Open file: {queue_path}")
            print("2) Edit labels/confidence if needed")
            if not args.auto_confirm:
                input("Press Enter after manual review is completed... ")

        if label_source_col:
            lineage_payload = {
                "label_source_method": label_source_method,
                "label_source_column": label_source_col,
                "allow_copy_from_existing_label": allow_copy_from_existing,
            }
            lineage_path.write_text(json.dumps(lineage_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    queue_path = paths.data / "review_queue.csv"
    if not auto_path.exists():
        raise FileNotFoundError(f"Missing auto-labeled file: {auto_path}")
    if not queue_path.exists():
        raise FileNotFoundError(f"Missing review queue file: {queue_path}")

    labeled_path = paths.data / "labeled_dataset.csv"
    queue_rows = max(0, sum(1 for _ in queue_path.open("r", encoding="utf-8")) - 1)
    reviewed_subset_metrics: dict[str, float | None] = {"kappa": None, "agreement": None}
    if queue_rows > 0:
        auto_df = pd.read_csv(auto_path)
        reviewed_df = pd.read_csv(queue_path)
        if "__row_id" in auto_df.columns and "__row_id" in reviewed_df.columns:
            base = auto_df.set_index("__row_id")
            rev = reviewed_df.set_index("__row_id")
            overlap = base.index.intersection(rev.index)
            if len(overlap) > 0:
                human_col = "human_label" if "human_label" in rev.columns else "auto_label"
                eval_df = pd.DataFrame(
                    {
                        "auto_label": base.loc[overlap, "auto_label"].astype(str),
                        "human_label": rev.loc[overlap, human_col].astype(str),
                    }
                ).reset_index(drop=True)
                m = calculate_metrics(eval_df, auto_col="auto_label", human_col="human_label", confidence_col="confidence")
                kappa_val = m.get("kappa")
                if kappa_val is not None and pd.isna(kappa_val):
                    kappa_val = None
                reviewed_subset_metrics["kappa"] = kappa_val  # type: ignore[assignment]
                reviewed_subset_metrics["agreement"] = float(
                    (eval_df["auto_label"].astype(str) == eval_df["human_label"].astype(str)).mean()
                )

        merge_reviewed(auto_path, queue_path, labeled_path, row_id_col="__row_id", label_col="auto_label")
    else:
        shutil.copy2(auto_path, labeled_path)

    class_defs = acfg.get("class_defs") or {}
    if not class_defs:
        labels = sorted(pd.read_csv(labeled_path)["auto_label"].dropna().astype(str).unique().tolist())
        class_defs = {label: "auto-generated" for label in labels}

    generate_spec(
        df=pd.read_csv(labeled_path),
        task_desc=cfg.get("task_description", "Annotation task"),
        class_defs=class_defs,
        output_path=paths.labeling / "annotation_spec.md",
        label_col="auto_label",
    )

    metrics = calculate_metrics(pd.read_csv(labeled_path), auto_col="auto_label", human_col=None, confidence_col="confidence")
    metrics["kappa"] = reviewed_subset_metrics.get("kappa")
    metrics["agreement"] = reviewed_subset_metrics.get("agreement")
    metrics["review_queue_rows"] = int(queue_rows)
    (paths.reports / "annotation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    display_cols = acfg.get("display_cols") or [c for c in pd.read_csv(labeled_path).columns if c not in {"reason"}]
    export_labelstudio(
        df=pd.read_csv(labeled_path),
        output_path=paths.labeling / "labelstudio_import.json",
        display_cols=display_cols,
        label_col="auto_label",
        confidence_col="confidence",
    )


def _stage_active_learning(cfg: dict, paths: RunPaths) -> None:
    labeled = paths.data / "labeled_dataset.csv"
    if not labeled.exists():
        raise FileNotFoundError(f"Missing input for AL stage: {labeled}")

    acfg = cfg.get("active_learning") or {}
    target_col = ((cfg.get("schema") or {}).get("target_col") or "auto_label")
    drop_cols = {"__row_id", "confidence", "reason", "collected_at", "source"}

    lineage_path = paths.reports / "annotation_lineage.json"
    if lineage_path.exists():
        try:
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            label_source_column = lineage.get("label_source_column")
            if label_source_column:
                drop_cols.add(str(label_source_column))
        except Exception:
            pass

    run_experiment(
        labeled_csv=labeled,
        reports_dir=paths.reports,
        models_dir=paths.models,
        target_col=target_col,
        drop_cols=sorted(drop_cols),
        n_start=int(acfg.get("n_start", 50)),
        iterations=int(acfg.get("iterations", 5)),
        batch_size=int(acfg.get("batch_size", 20)),
        test_size=float(acfg.get("test_size", 0.2)),
        random_state=int(acfg.get("random_state", 42)),
    )


def _stage_final_report(paths: RunPaths) -> None:
    build_report(paths.root, paths.root / "README.md")
    build_report_html(paths.root, paths.reports / "final_report.html")


def _build_stage_map() -> dict[str, Callable]:
    return {
        "data_collection": lambda cfg, paths, args: _stage_data_collection(cfg, paths, args),
        "data_quality": _stage_data_quality,
        "annotation": _stage_annotation,
        "active_learning": lambda cfg, paths, args: _stage_active_learning(cfg, paths),
        "final_report": lambda cfg, paths, args: _stage_final_report(paths),
    }


def _select_stage_slice(from_stage: str | None, to_stage: str | None, only_stage: str | None) -> list[str]:
    if only_stage:
        if only_stage not in STAGES:
            raise ValueError(f"Unknown stage: {only_stage}")
        return [only_stage]

    start_idx = STAGES.index(from_stage) if from_stage else 0
    end_idx = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
    if end_idx < start_idx:
        raise ValueError("to_stage must not be before from_stage")
    return STAGES[start_idx : end_idx + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-entry orchestrator for full Codex pipeline")
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--topic", default=None, help="Topic/domain for auto dataset discovery")
    parser.add_argument("--run_root", default=None, help="Use existing run root; if omitted, a new run is created")
    parser.add_argument("--auto_confirm", action="store_true", help="Skip stage transition prompts")
    parser.add_argument("--no_review_pause", action="store_true", help="Do not pause for manual review queue editing")
    parser.add_argument("--quality_strategy", choices=["aggressive", "smart", "conservative"], default=None)
    parser.add_argument("--strategy_rationale", default=None)
    parser.add_argument("--annotation_mode", choices=["full", "queue_only", "merge_only"], default="full")
    parser.add_argument("--from_stage", choices=STAGES, default=None)
    parser.add_argument("--to_stage", choices=STAGES, default=None)
    parser.add_argument("--only_stage", choices=STAGES, default=None)
    args = parser.parse_args()

    cfg = _load_cfg()

    if args.run_root:
        paths = _ensure_run_paths(Path(args.run_root).resolve())
    else:
        task_name = args.task_name or cfg.get("task_name") or (args.topic or "task")
        paths = make_run_folder(PROJECT_ROOT, task_name)
        snapshot = {
            "task_name": task_name,
            "task_description": cfg.get("task_description", ""),
            "topic": args.topic,
            "pipeline_mode": "single_entry_orchestrator",
        }
        (paths.root / "run_config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Run root: {paths.root}")

    stage_map = _build_stage_map()
    selected_stages = _select_stage_slice(args.from_stage, args.to_stage, args.only_stage)

    for stage in selected_stages:
        print(f"\n=== Stage: {stage} ===")
        stage_map[stage](cfg, paths, args)
        _checkpoint(stage, args.auto_confirm)

    print("\nPipeline finished.")
    print(f"Artifacts: {paths.root}")


if __name__ == "__main__":
    main()
