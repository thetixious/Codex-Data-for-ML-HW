from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _load_local_env_file() -> None:
    env_path = PROJECT_ROOT / ".envar"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _prepare_kaggle_auth() -> None:
    _load_local_env_file()
    token = os.getenv("KAGGLE_API_TOKEN")
    if token:
        token = token.strip()
        if token.startswith("{"):
            try:
                payload = json.loads(token)
                if payload.get("key") and not os.getenv("KAGGLE_KEY"):
                    os.environ["KAGGLE_KEY"] = str(payload["key"])
                if payload.get("username") and not os.getenv("KAGGLE_USERNAME"):
                    os.environ["KAGGLE_USERNAME"] = str(payload["username"])
            except Exception:
                pass
        elif ":" in token and not os.getenv("KAGGLE_USERNAME"):
            user, key = token.split(":", 1)
            if user and key:
                os.environ.setdefault("KAGGLE_USERNAME", user.strip())
                os.environ.setdefault("KAGGLE_KEY", key.strip())
        elif not os.getenv("KAGGLE_KEY"):
            os.environ["KAGGLE_KEY"] = token


def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> Path:
    _prepare_kaggle_auth()
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Set KAGGLE_USERNAME + KAGGLE_KEY "
            "or provide KAGGLE_API_TOKEN (JSON with username/key, 'username:key', or key with username env)."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kaggle_dl_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        api.dataset_download_files(dataset_name, path=str(tmp_path), unzip=False)

        zip_files = list(tmp_path.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError("Kaggle zip archive not found after download")

        csv_candidates: list[Path] = []
        for zf in zip_files:
            with zipfile.ZipFile(zf, "r") as archive:
                archive.extractall(tmp_path / zf.stem)
            csv_candidates.extend((tmp_path / zf.stem).rglob("*.csv"))

        if not csv_candidates:
            raise FileNotFoundError("No CSV files inside Kaggle archive")

        biggest = max(csv_candidates, key=lambda p: p.stat().st_size)
        out_path = output_dir / f"kaggle_{safe_name(dataset_name)}.csv"
        shutil.copy2(biggest, out_path)

    logger.info("Saved Kaggle dataset -> %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle dataset to CSV")
    parser.add_argument("--name", required=True, help="Kaggle dataset id: owner/dataset")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    path = download_kaggle_dataset(args.name, Path(args.output_dir))
    print(path)


if __name__ == "__main__":
    main()
