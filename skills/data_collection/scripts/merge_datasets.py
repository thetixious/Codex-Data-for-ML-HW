from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def merge_csv_files(input_dir: Path, output_file: Path, pattern: str = "*.csv") -> Path:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for pattern '{pattern}' in {input_dir}")

    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except Exception as exc:
            logger.warning("Skipping %s: %s", file, exc)

    if not frames:
        raise RuntimeError("No readable files to merge")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    logger.info("Merged %s files -> %s (rows=%s)", len(frames), output_file, len(merged))
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CSV files from directory")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--pattern", default="*.csv")
    args = parser.parse_args()

    path = merge_csv_files(Path(args.input_dir), Path(args.output_file), args.pattern)
    print(path)


if __name__ == "__main__":
    main()
