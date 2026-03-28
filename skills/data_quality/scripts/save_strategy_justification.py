from __future__ import annotations

import argparse
from pathlib import Path


def save_justification(strategy: str, rationale: str, output_path: Path) -> Path:
    text = (
        "# Strategy Justification\n\n"
        f"Selected strategy: **{strategy}**\n\n"
        "## Rationale\n"
        f"{rationale.strip()}\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Save user-approved cleaning strategy rationale")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = save_justification(args.strategy, args.rationale, Path(args.output))
    print(out)


if __name__ == "__main__":
    main()
