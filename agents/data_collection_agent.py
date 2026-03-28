from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any

from bs4 import BeautifulSoup
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import requests
import yaml


STANDARD_COLUMNS = ["text", "audio", "image", "label", "source", "collected_at"]
TEXT_HINTS = {"text", "comment", "review", "content", "body", "message", "title", "question", "prompt"}
LABEL_HINTS = {"label", "target", "class", "category", "sentiment", "intent", "topic", "tag", "y"}
AUDIO_HINTS = {"audio", "wave", "wav", "mp3", "sound", "recording"}
IMAGE_HINTS = {"image", "img", "photo", "picture", "jpg", "jpeg", "png"}


@dataclass
class AgentConfig:
    output_raw_dir: Path
    notebooks_dir: Path


class DataCollectionAgent:
    def __init__(self, config: str | Path = "config.yaml") -> None:
        cfg_path = Path(config)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

        output_raw = ((cfg.get("paths") or {}).get("raw_data_dir")) or "data/raw"
        notebooks = ((cfg.get("paths") or {}).get("notebooks_dir")) or "notebooks"

        self.config = AgentConfig(
            output_raw_dir=Path(output_raw),
            notebooks_dir=Path(notebooks),
        )

        self.config.output_raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.notebooks_dir.mkdir(parents=True, exist_ok=True)

    def scrape(self, url: str, selector: str) -> pd.DataFrame:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        nodes = soup.select(selector)
        rows = [{"text": n.get_text(" ", strip=True)} for n in nodes if n.get_text(strip=True)]
        return pd.DataFrame(rows)

    def fetch_api(self, endpoint: str, params: dict | None = None) -> pd.DataFrame:
        req_params = dict(params or {})
        attempts: list[dict[str, Any]] = [req_params]

        no_type = dict(req_params)
        no_type.pop("type", None)
        if no_type != req_params:
            attempts.append(no_type)

        no_size = dict(no_type)
        no_size.pop("size", None)
        if no_size not in attempts:
            attempts.append(no_size)

        if "q" in req_params:
            q_only = {"q": req_params["q"]}
            if q_only not in attempts:
                attempts.append(q_only)

        if {} not in attempts:
            attempts.append({})

        last_error: requests.HTTPError | None = None
        resp = None
        for candidate in attempts:
            resp = requests.get(endpoint, params=candidate, timeout=30)
            try:
                resp.raise_for_status()
                break
            except requests.HTTPError as err:
                last_error = err
                continue

        if resp is None or (last_error is not None and resp.status_code >= 400):
            assert last_error is not None
            raise last_error
        payload = resp.json()

        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            for key in ("data", "results", "items", "records"):
                if isinstance(payload.get(key), list):
                    return pd.json_normalize(payload[key])
            hits = payload.get("hits")
            if isinstance(hits, dict) and isinstance(hits.get("hits"), list):
                return pd.json_normalize(hits["hits"])
            return pd.json_normalize([payload])

        raise ValueError("Unsupported API payload type")

    def load_dataset(self, name: str, source: str = "hf") -> pd.DataFrame:
        source = source.lower()
        if source == "hf":
            ds = load_dataset(name)
            split = "train" if "train" in ds else next(iter(ds.keys()))
            return ds[split].to_pandas()

        if source == "kaggle":
            api = KaggleApi()
            api.authenticate()
            with tempfile.TemporaryDirectory(prefix="kaggle_agent_") as tmp:
                tmp_path = Path(tmp)
                api.dataset_download_files(name, path=str(tmp_path), unzip=True)
                csv_files = sorted(tmp_path.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
                if not csv_files:
                    raise FileNotFoundError("No CSV files in Kaggle dataset")
                return pd.read_csv(csv_files[0])

        raise ValueError("source must be 'hf' or 'kaggle'")

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        unified = [self._to_standard_schema(df) for df in sources]
        out = pd.concat(unified, ignore_index=True)
        out = out.drop_duplicates().reset_index(drop=True)
        return out

    def run(self, sources: list[dict[str, Any]]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for src in sources:
            stype = src.get("type")
            if stype == "hf_dataset":
                df = self.load_dataset(src["name"], source="hf")
                df["source"] = f"hf:{src['name']}"
            elif stype == "kaggle_dataset":
                df = self.load_dataset(src["name"], source="kaggle")
                df["source"] = f"kaggle:{src['name']}"
            elif stype == "scrape":
                df = self.scrape(src["url"], src["selector"])
                df["source"] = src.get("source_name", "scrape")
            elif stype == "api":
                df = self.fetch_api(src["endpoint"], src.get("params"))
                df["source"] = src.get("source_name", "api")
            else:
                raise ValueError(f"Unsupported source type: {stype}")

            frames.append(df)

        merged = self.merge(frames)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_csv = self.config.output_raw_dir / f"collected_{ts}.csv"
        merged.to_csv(out_csv, index=False)

        self._build_eda_notebook(merged, self.config.notebooks_dir / "eda.ipynb")
        return merged

    def _to_standard_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        now = datetime.now().isoformat(timespec="seconds")

        if "source" not in work.columns:
            work["source"] = "unknown"
        if "collected_at" not in work.columns:
            work["collected_at"] = now

        text_col = self._find_col(work, TEXT_HINTS)
        audio_col = self._find_col(work, AUDIO_HINTS)
        image_col = self._find_col(work, IMAGE_HINTS)
        label_col = self._find_col(work, LABEL_HINTS)

        out = pd.DataFrame(index=work.index)
        out["text"] = work[text_col].astype(str) if text_col else ""
        out["audio"] = work[audio_col].astype(str) if audio_col else ""
        out["image"] = work[image_col].astype(str) if image_col else ""

        if label_col:
            out["label"] = work[label_col].astype(str)
        else:
            # fallback: weak pseudo label for unlabeled data
            out["label"] = "unlabeled"

        out["source"] = work["source"].astype(str)
        out["collected_at"] = work["collected_at"].astype(str)

        return out[STANDARD_COLUMNS]

    @staticmethod
    def _find_col(df: pd.DataFrame, hints: set[str]) -> str | None:
        candidates = []
        for col in df.columns:
            norm = "".join(ch.lower() if ch.isalnum() else "_" for ch in col)
            parts = {p for p in norm.split("_") if p}
            if parts & hints:
                candidates.append(col)
        return candidates[0] if candidates else None

    def _build_eda_notebook(self, df: pd.DataFrame, out_path: Path) -> None:
        latest = sorted(self.config.output_raw_dir.glob("collected_*.csv"))
        csv_path = (self.config.output_raw_dir / latest[-1].name).as_posix() if latest else (self.config.output_raw_dir / "collected_unified.csv").as_posix()

        def md(text: str) -> dict:
            return {"cell_type": "markdown", "metadata": {}, "source": text}

        def code(text: str) -> dict:
            return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}

        nb = {
            "cells": [
                md("# EDA: Unified Dataset"),
                code(
                    "import pandas as pd\n"
                    "import matplotlib.pyplot as plt\n"
                    "from collections import Counter\n"
                    f"df = pd.read_csv(r'{csv_path}')\n"
                    "df.head()"
                ),
                md("## Class Distribution"),
                code(
                    "ax = df['label'].value_counts().plot(kind='bar', figsize=(8,4), title='Class Distribution')\n"
                    "ax.set_xlabel('label')\n"
                    "ax.set_ylabel('count')\n"
                    "plt.tight_layout()"
                ),
                md("## Text Length Distribution"),
                code(
                    "txt = df['text'].fillna('')\n"
                    "lengths = txt.str.len()\n"
                    "lengths.describe()"
                ),
                code(
                    "lengths.hist(bins=40, figsize=(8,4))\n"
                    "plt.title('Text length distribution')\n"
                    "plt.xlabel('chars')\n"
                    "plt.ylabel('count')\n"
                    "plt.tight_layout()"
                ),
                md("## Top-20 Words"),
                code(
                    "words = ' '.join(df['text'].fillna('').astype(str)).lower().split()\n"
                    "top20 = Counter([w for w in words if len(w) > 2]).most_common(20)\n"
                    "pd.DataFrame(top20, columns=['word','count'])"
                ),
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["DataCollectionAgent"]
