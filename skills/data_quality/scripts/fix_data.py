from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


STRATEGY_ALIASES = {
    "strict": "aggressive",
    "medium": "smart",
    "mild": "conservative",
}


def _normalize_strategy(strategy: str) -> str:
    return STRATEGY_ALIASES.get(strategy, strategy)


def _clip_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return df[col].clip(lower=low, upper=high)


def fix_data(input_csv: Path, output_csv: Path, strategy: str) -> Path:
    strategy = _normalize_strategy(strategy)
    if strategy not in {"aggressive", "smart", "conservative"}:
        raise ValueError("Strategy must be one of: aggressive, smart, conservative")

    df = pd.read_csv(input_csv)
    out = df.drop_duplicates().copy()

    numeric_cols = out.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if strategy == "aggressive":
        out = out.dropna().reset_index(drop=True)
        for col in numeric_cols:
            out[col] = _clip_iqr(out, col)

    elif strategy == "conservative":
        for col in numeric_cols:
            out[col] = out[col].fillna(out[col].median())
        for col in categorical_cols:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])

    elif strategy == "smart":
        for col in numeric_cols:
            out[col] = out[col].fillna(out[col].median())
            mean = out[col].mean()
            std = out[col].std(ddof=0)
            if std > 0:
                out = out[(out[col] >= mean - 3 * std) & (out[col] <= mean + 3 * std)]
        for col in categorical_cols:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])

    out = out.reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    logger.info("Cleaned data with '%s' strategy: %s -> %s (rows=%s)", strategy, input_csv, output_csv, len(out))
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean CSV with selected strategy")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--strategy", required=True, choices=["aggressive", "smart", "conservative", "strict", "medium", "mild"])
    args = parser.parse_args()

    out = fix_data(Path(args.input_csv), Path(args.output_csv), args.strategy)
    print(out)


if __name__ == "__main__":
    main()
