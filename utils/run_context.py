from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


@dataclass
class RunPaths:
    root: Path
    data: Path
    reports: Path
    models: Path
    notebooks: Path
    labeling: Path
    scripts: Path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "task"


def make_run_folder(project_root: Path, task_name: str, now: datetime | None = None) -> RunPaths:
    now = now or datetime.now()
    run_name = f"{slugify(task_name)}_{now.strftime('%Y-%m-%d_%H-%M')}"
    root = project_root / "data" / "raw" / run_name

    paths = RunPaths(
        root=root,
        data=root / "data",
        reports=root / "reports",
        models=root / "models",
        notebooks=root / "notebooks",
        labeling=root / "labeling",
        scripts=root / "scripts",
    )
    for p in paths.__dict__.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return paths


def parse_run_root_arg() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run_root", required=True, help="Path to run folder root")
    return parser
