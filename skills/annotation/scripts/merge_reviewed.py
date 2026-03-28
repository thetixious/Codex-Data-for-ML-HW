from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_reviewed(
    auto_labeled_csv: Path,
    reviewed_csv: Path,
    output_csv: Path,
    row_id_col: str = "__row_id",
    label_col: str = "auto_label",
) -> Path:
    base = pd.read_csv(auto_labeled_csv)
    reviewed = pd.read_csv(reviewed_csv)

    if row_id_col not in base.columns or row_id_col not in reviewed.columns:
        raise KeyError(f"Both files must contain '{row_id_col}'")
    if label_col not in reviewed.columns:
        raise KeyError(f"Reviewed file must contain '{label_col}'")

    base = base.set_index(row_id_col)
    reviewed = reviewed.set_index(row_id_col)

    overlap = base.index.intersection(reviewed.index)
    if len(overlap) == 0:
        raise ValueError("No overlapping row ids between base and reviewed files")

    updatable_cols = [c for c in reviewed.columns if c in base.columns]
    for col in updatable_cols:
        right = reviewed.loc[overlap, col]
        left_dtype = base[col].dtype
        try:
            if str(left_dtype).startswith("string") or str(left_dtype) == "object":
                right = right.astype(str)
            else:
                right = right.astype(left_dtype, copy=False)
        except Exception:
            pass
        base.loc[overlap, col] = right

    merged = base.reset_index().sort_values(row_id_col).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge reviewed queue into auto-labeled dataset")
    parser.add_argument("--auto_labeled_csv", required=True)
    parser.add_argument("--reviewed_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--row_id_col", default="__row_id")
    parser.add_argument("--label_col", default="auto_label")
    args = parser.parse_args()

    out = merge_reviewed(
        auto_labeled_csv=Path(args.auto_labeled_csv),
        reviewed_csv=Path(args.reviewed_csv),
        output_csv=Path(args.output_csv),
        row_id_col=args.row_id_col,
        label_col=args.label_col,
    )
    print(out)


if __name__ == "__main__":
    main()
