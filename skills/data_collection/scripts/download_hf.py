from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from datasets import load_dataset
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def download_hf_dataset(dataset_name: str, output_dir: Path, subset: str | None, split: str | None) -> Path:
    load_dotenv()
    token = None
    # datasets lib also reads HF_TOKEN/HF_HOME automatically.

    logger.info("Loading dataset '%s' from Hugging Face", dataset_name)
    ds = load_dataset(dataset_name, subset, token=token, trust_remote_code=True) if subset else load_dataset(dataset_name, token=token, trust_remote_code=True)

    if split is None:
        split = "train" if "train" in ds else next(iter(ds.keys()))
    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    df = ds[split].to_pandas()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{safe_name(subset)}" if subset else ""
    filename = f"hf_{safe_name(dataset_name)}{suffix}_{safe_name(split)}.csv"
    output_path = output_dir / filename

    df.to_csv(output_path, index=False)
    logger.info("Saved %s rows -> %s", len(df), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset split to CSV")
    parser.add_argument("--name", required=True, help="HF dataset id, e.g. imdb")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None, help="train/test/validation or custom")
    args = parser.parse_args()

    path = download_hf_dataset(args.name, Path(args.output_dir), args.subset, args.split)
    print(path)


if __name__ == "__main__":
    main()
