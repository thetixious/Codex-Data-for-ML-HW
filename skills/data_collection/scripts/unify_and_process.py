from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def unify_dataframe(
    input_path: Path,
    output_path: Path,
    rename_map: dict[str, str],
    keep_cols: list[str],
    source_name: str,
    run_timestamp: str,
    drop_duplicates: bool = True,
) -> Path:
    df = pd.read_csv(input_path)

    valid_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_map)

    if keep_cols:
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols].copy()

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    df["source"] = source_name
    df["collected_at"] = run_timestamp

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Unified %s -> %s (rows=%s)", input_path, output_path, len(df))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify dataset to fixed schema")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--rename_map", required=True, help="JSON dict old->new")
    parser.add_argument("--keep_cols", required=True, help="JSON list of columns to keep")
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--run_timestamp", required=True)
    parser.add_argument("--no_drop_duplicates", action="store_true")
    args = parser.parse_args()

    rename_map = json.loads(args.rename_map)
    keep_cols = json.loads(args.keep_cols)

    path = unify_dataframe(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        rename_map=rename_map,
        keep_cols=keep_cols,
        source_name=args.source_name,
        run_timestamp=args.run_timestamp,
        drop_duplicates=not args.no_drop_duplicates,
    )
    print(path)


if __name__ == "__main__":
    main()
