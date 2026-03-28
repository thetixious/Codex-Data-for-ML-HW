from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import requests

try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None  # type: ignore

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None  # type: ignore


STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "on", "in", "with", "by", "from", "data", "dataset",
    "и", "в", "на", "для", "по", "из", "с", "к", "о", "об", "под", "данные", "датасет",
}
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_local_env_file() -> None:
    env_path = PROJECT_ROOT / ".envar"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass
class DatasetCandidate:
    source: str
    dataset_id: str
    name: str
    url: str
    description: str
    relevance_score: float


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def _score(topic: str, title: str, description: str) -> float:
    t_topic = _tokens(topic)
    t_desc = _tokens(f"{title} {description}")
    if not t_topic:
        return 1.0

    overlap = len(t_topic & t_desc) / max(1, len(t_topic))
    phrase_bonus = 0.25 if topic.lower() in f"{title} {description}".lower() else 0.0
    rich_bonus = min(0.25, math.log1p(len(t_desc)) / 20)

    score = 10.0 * min(1.0, overlap + phrase_bonus + rich_bonus)
    return round(max(0.1, score), 2)


def discover_huggingface(topic: str, limit: int = 20) -> list[DatasetCandidate]:
    if HfApi is None:
        return []

    api = HfApi()
    items = []

    try:
        for ds in api.list_datasets(search=topic, limit=limit, full=False):
            ds_id = getattr(ds, "id", None)
            if not ds_id:
                continue

            tags = getattr(ds, "tags", None) or []
            desc = " ".join(str(t) for t in tags[:20])
            name = ds_id.split("/")[-1]
            score = _score(topic, ds_id, desc)

            items.append(
                DatasetCandidate(
                    source="hf",
                    dataset_id=ds_id,
                    name=name,
                    url=f"https://huggingface.co/datasets/{ds_id}",
                    description=desc,
                    relevance_score=score,
                )
            )
    except Exception:
        return []

    return items


def discover_kaggle(topic: str, limit: int = 20) -> list[DatasetCandidate]:
    if KaggleApi is None:
        return []

    _load_local_env_file()
    api = KaggleApi()
    items = []

    try:
        api.authenticate()
        datasets = api.dataset_list(search=topic, page_size=limit)
        for ds in datasets:
            ds_ref = getattr(ds, "ref", None)
            if not ds_ref:
                continue
            title = getattr(ds, "title", "") or ds_ref
            subtitle = getattr(ds, "subtitle", "") or ""
            score = _score(topic, title, subtitle)

            items.append(
                DatasetCandidate(
                    source="kaggle",
                    dataset_id=ds_ref,
                    name=title,
                    url=f"https://www.kaggle.com/datasets/{ds_ref}",
                    description=subtitle,
                    relevance_score=score,
                )
            )
    except Exception:
        return []

    return items


def discover_zenodo(topic: str, limit: int = 20) -> list[DatasetCandidate]:
    items = []
    try:
        resp = requests.get(
            "https://zenodo.org/api/records",
            params={"q": topic, "type": "dataset", "size": limit, "sort": "mostviewed"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        hits = (data.get("hits") or {}).get("hits") or []
        for hit in hits:
            rid = str(hit.get("id", ""))
            meta = hit.get("metadata") or {}
            title = meta.get("title", "") or rid
            desc = meta.get("description", "") or ""
            score = _score(topic, title, desc)

            items.append(
                DatasetCandidate(
                    source="zenodo",
                    dataset_id=rid,
                    name=title,
                    url=(hit.get("links") or {}).get("html", f"https://zenodo.org/records/{rid}"),
                    description=re.sub(r"<[^>]+>", " ", desc)[:500],
                    relevance_score=score,
                )
            )
    except Exception:
        return []

    return items


def discover_datasets(topic: str, limit_per_source: int = 20, sources: Iterable[str] = ("hf", "kaggle", "zenodo")) -> list[DatasetCandidate]:
    out: list[DatasetCandidate] = []
    src_set = set(sources)

    if "hf" in src_set:
        out.extend(discover_huggingface(topic, limit_per_source))
    if "kaggle" in src_set:
        out.extend(discover_kaggle(topic, limit_per_source))
    if "zenodo" in src_set:
        out.extend(discover_zenodo(topic, limit_per_source))

    dedup: dict[tuple[str, str], DatasetCandidate] = {}
    for item in out:
        key = (item.source, item.dataset_id)
        prev = dedup.get(key)
        if prev is None or item.relevance_score > prev.relevance_score:
            dedup[key] = item

    merged = sorted(dedup.values(), key=lambda x: x.relevance_score, reverse=True)
    return merged


def select_candidates(candidates: list[DatasetCandidate], top_k: int = 4, min_score: float = 2.0) -> list[DatasetCandidate]:
    filtered = [c for c in candidates if c.relevance_score >= min_score]
    if not filtered:
        return candidates[:top_k]

    selected: list[DatasetCandidate] = []
    used_sources: set[str] = set()

    for c in filtered:
        if c.source not in used_sources:
            selected.append(c)
            used_sources.add(c.source)
        if len(selected) >= top_k:
            return selected

    for c in filtered:
        if c in selected:
            continue
        selected.append(c)
        if len(selected) >= top_k:
            break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover dataset candidates for a topic")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--limit_per_source", type=int, default=20)
    parser.add_argument("--sources", default="hf,kaggle,zenodo", help="Comma-separated: hf,kaggle,zenodo")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--min_score", type=float, default=2.0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    candidates = discover_datasets(args.topic, args.limit_per_source, sources)
    selected = select_candidates(candidates, args.top_k, args.min_score)

    payload = {
        "topic": args.topic,
        "sources": sources,
        "total_candidates": len(candidates),
        "candidates": [asdict(c) for c in candidates],
        "selected": [asdict(c) for c in selected],
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
