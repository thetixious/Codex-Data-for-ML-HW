from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.run_context import make_run_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated run directory")
    parser.add_argument("--task_name", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    task_name = args.task_name or cfg.get("task_name") or "task"

    run_paths = make_run_folder(PROJECT_ROOT, task_name)

    # Store task snapshot
    snapshot = {
        "task_name": task_name,
        "task_description": cfg.get("task_description", ""),
    }
    (run_paths.root / "run_config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(run_paths.root)


if __name__ == "__main__":
    main()
