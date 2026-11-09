"""Helpers for the Simuletic CCTV smoke/fire dataset: locate files, parse names, remap classes."""
import re
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png"}
# known-bad files, listed in params.yaml (data.simuletic_exclude) so the exclusion is explicit
EXCLUDE = set(yaml.safe_load(open("params.yaml"))["data"].get("simuletic_exclude") or [])


def find_root(raw):
    hits = sorted(Path(raw).rglob("data.yaml"))
    if not hits:
        raise SystemExit(f"no data.yaml found under {raw} - run the ingest stage first")
    return hits[0].parent


def source_names(root):
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def remap_table(src_names, classes_cfg):
    """Map Simuletic class ids to our ids by NAME, so the source order never matters."""
    name2id = {v: int(k) for k, v in classes_cfg.items()}
    table = {}
    for sid, name in src_names.items():
        hits = [c for c in name2id if c in str(name).lower()]
        if len(hits) != 1:
            raise SystemExit(f"cannot map Simuletic class {sid}='{name}' to {list(name2id)}; "
                             f"source names: {src_names}")
        table[sid] = name2id[hits[0]]
    return table


def variant_of(stem):
    m = re.search(r"var(\d+)", stem)
    return int(m.group(1)) if m else None


def label_path(root, img):
    parts = list(img.relative_to(root).parts)
    idx = len(parts) - 1 - parts[::-1].index("images")  # last 'images' folder in the path
    parts[idx] = "labels"
    return (root / Path(*parts)).with_suffix(".txt")


def list_items(root):
    items = []
    for img in sorted(root.rglob("*")):
        if img.stem in EXCLUDE:
            continue
        if img.suffix.lower() in IMG_EXT and "images" in img.relative_to(root).parts:
            items.append({"img": img, "lbl": label_path(root, img), "stem": img.stem,
                          "variant": variant_of(img.stem)})
    return items