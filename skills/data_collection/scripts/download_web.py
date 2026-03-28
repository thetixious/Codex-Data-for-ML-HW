from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import urlparse
import uuid

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def load_max_size_mb() -> int:
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        return 300
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return int((cfg.get("download") or {}).get("max_file_size_mb", 300))


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def download_web_file(url: str, output_dir: Path) -> Path:
    max_size_mb = load_max_size_mb()
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        length = response.headers.get("content-length")
        if length is not None:
            size_mb = int(length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"File too large: {size_mb:.1f} MB > {max_size_mb} MB")

        content_type = (response.headers.get("content-type") or "").lower()
        if "text/html" in content_type:
            raise ValueError("URL points to HTML page, expected raw file")

        parsed = urlparse(url)
        name = Path(parsed.path).name
        if not name:
            ext = ".csv" if "csv" in content_type else ".bin"
            name = f"web_{uuid.uuid4().hex[:12]}{ext}"

        out_path = output_dir / f"web_{safe_name(name)}"
        with out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)

    logger.info("Saved web file -> %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw file from direct URL")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    path = download_web_file(args.url, Path(args.output_dir))
    print(path)


if __name__ == "__main__":
    main()
