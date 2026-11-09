"""Stage ingest: make sure D-Fire is present in data.raw_dir.

Downloads from Kaggle only when the data is missing or incomplete, then verifies
the image counts per split. Needs the kaggle CLI (venv active) and ~/.kaggle/kaggle.json.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_SLUG = "sayedgamal99/smoke-fire-detection-yolo"
DEFAULT_COUNTS = {"train": 14122, "val": 3099, "test": 4306}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def count_images(raw, split):
    d = Path(raw) / "data" / split / "images"
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else 0


def counts(raw, splits):
    return {s: count_images(raw, s) for s in splits}


def download(slug, raw, attempts=3):
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        sys.exit("kaggle CLI not found: activate the venv (pip install kaggle) and put "
                 "kaggle.json in your .kaggle folder")
    Path(raw).mkdir(parents=True, exist_ok=True)
    for i in range(1, attempts + 1):
        r = subprocess.run([kaggle, "datasets", "download", "-d", slug, "-p", str(raw), "--unzip"])
        if r.returncode == 0:
            return
        print(f"kaggle download failed (attempt {i}/{attempts})")
    sys.exit("download failed after retries - check your connection and kaggle.json")


def main():
    P = yaml.safe_load(open("params.yaml"))["data"]
    raw = Path(P["raw_dir"])
    slug = P.get("kaggle_dataset", DEFAULT_SLUG)
    expected = P.get("expected_counts", DEFAULT_COUNTS)

    have = counts(raw, expected)
    if have != expected:
        print(f"D-Fire missing or incomplete in {raw} ({have}); downloading {slug}")
        download(slug, raw)
        have = counts(raw, expected)
    if have != expected:
        sys.exit(f"ingest check failed: expected {expected}, found {have}")
    print(json.dumps({"dataset": slug, "dir": str(raw), "images": have}, indent=2))


if __name__ == "__main__":
    main()