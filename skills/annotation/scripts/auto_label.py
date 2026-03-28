from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def _match_rule(value: Any, rule: dict[str, Any]) -> tuple[bool, str]:
    rtype = rule.get("type")

    if rtype == "threshold":
        op = rule.get("op")
        threshold = float(rule["val"])
        try:
            v = float(value)
        except Exception:
            return False, "not_numeric"
        if op == "<":
            return v < threshold, f"{v} < {threshold}"
        if op == "<=":
            return v <= threshold, f"{v} <= {threshold}"
        if op == ">":
            return v > threshold, f"{v} > {threshold}"
        if op == ">=":
            return v >= threshold, f"{v} >= {threshold}"
        return False, "unsupported_op"

    if rtype == "range":
        try:
            v = float(value)
        except Exception:
            return False, "not_numeric"
        low = float(rule.get("min", float("-inf")))
        high = float(rule.get("max", float("inf")))
        match = low <= v < high
        return match, f"{low} <= {v} < {high}"

    if rtype == "keyword":
        text = str(value).lower()
        token = str(rule.get("val", "")).lower()
        return token in text, f"contains('{token}')"

    if rtype == "regex":
        text = str(value)
        pattern = str(rule.get("pattern", ""))
        match = re.search(pattern, text, flags=re.IGNORECASE) is not None
        return match, f"regex('{pattern}')"

    if rtype == "default":
        return True, "default"

    return False, "unsupported_rule"


def apply_rule(value: Any, rules: list[dict[str, Any]], fallback_label: str = "Unknown", fallback_conf: float = 0.5) -> tuple[str, float, str]:
    for i, rule in enumerate(rules):
        matched, reason = _match_rule(value, rule)
        if matched:
            return str(rule.get("label", fallback_label)), float(rule.get("conf", fallback_conf)), f"rule_{i}:{reason}"
    return fallback_label, fallback_conf, "fallback"


def auto_label(input_csv: Path, output_csv: Path, column: str, rules: list[dict[str, Any]], fallback_label: str = "Unknown", fallback_conf: float = 0.5) -> Path:
    df = pd.read_csv(input_csv)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    if "__row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["__row_id"] = df.index.astype(int)

    labels = df[column].apply(lambda value: apply_rule(value, rules, fallback_label=fallback_label, fallback_conf=fallback_conf))
    df["auto_label"] = labels.apply(lambda x: x[0])
    df["confidence"] = labels.apply(lambda x: x[1])
    df["reason"] = labels.apply(lambda x: x[2])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Auto-labeled dataset saved -> %s (rows=%s)", output_csv, len(df))
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based auto labeling")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--rules", required=True, help="JSON list of labeling rules")
    parser.add_argument("--fallback_label", default="Unknown")
    parser.add_argument("--fallback_conf", type=float, default=0.5)
    args = parser.parse_args()

    rules = json.loads(args.rules)
    out = auto_label(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        column=args.column,
        rules=rules,
        fallback_label=args.fallback_label,
        fallback_conf=args.fallback_conf,
    )
    print(out)


if __name__ == "__main__":
    main()
