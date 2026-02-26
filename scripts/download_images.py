"""Helper script to download WikiArt images from a URL manifest.

Reads all unique image paths from the six CSV files, checks which images are
already present on disk, and downloads any missing ones concurrently using a
thread pool.  Images are saved to ``<output_dir>/<StyleFolder>/<filename>``.

Usage::

    python scripts/download_images.py \\
        --data_dir data/wikiart \\
        --output data/wikiart/images \\
        --base_url https://your-host/wikiart \\
        --workers 8
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_CSV_FILES = [
    "style_train.csv", "style_val.csv",
    "genre_train.csv", "genre_val.csv",
    "artist_train.csv", "artist_val.csv",
]


def _collect_rel_paths(data_dir: pathlib.Path) -> List[str]:
    """Collect unique relative image paths from all CSVs."""
    rel_paths: set[str] = set()
    for csv_name in _CSV_FILES:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            logger.warning("CSV not found, skipping: %s", csv_path)
            continue
        with csv_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rel_path = line.rsplit(",", 1)[0]
                    rel_paths.add(rel_path)
    return sorted(rel_paths)


def _download_one(
    rel_path: str,
    base_url: str,
    output_dir: pathlib.Path,
    retries: int = 3,
    timeout: int = 30,
) -> Tuple[str, bool]:
    """Download a single image; returns (rel_path, success)."""
    dest = output_dir / rel_path
    if dest.exists():
        return rel_path, True  # already on disk

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{rel_path}"

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "DeepArtNet/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                dest.write_bytes(resp.read())
            return rel_path, True
        except (URLError, OSError) as exc:
            if attempt == retries:
                logger.error("Failed after %d attempts: %s — %s", retries, rel_path, exc)
                return rel_path, False
            time.sleep(2 ** attempt)  # exponential back-off

    return rel_path, False  # unreachable but satisfies type checker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WikiArt images")
    parser.add_argument("--data_dir", default="data/wikiart",
                        help="Directory containing the CSV files")
    parser.add_argument("--output", default="data/wikiart/images",
                        help="Destination image directory")
    parser.add_argument("--base_url", required=True,
                        help="Base URL prefix for images (e.g. https://host/wikiart)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent download threads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rel_paths = _collect_rel_paths(data_dir)
    missing = [p for p in rel_paths if not (output_dir / p).exists()]

    logger.info("Total images: %d | Already on disk: %d | To download: %d",
                len(rel_paths), len(rel_paths) - len(missing), len(missing))

    if not missing:
        logger.info("All images already present.")
        return

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, p, args.base_url, output_dir): p
            for p in missing
        }
        for i, future in enumerate(as_completed(futures), start=1):
            rel_path, ok = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
            if i % 500 == 0 or i == len(missing):
                logger.info("Progress: %d / %d (failed: %d)", i, len(missing), fail_count)

    logger.info("Download complete. Success: %d | Failed: %d", success_count, fail_count)


if __name__ == "__main__":
    main()
