"""Stage prepare_simuletic: remap classes, split BY VARIANT (no scene variant in two splits),
write data/processed/simuletic/ (+ baseline_labels.csv, data.yaml) and the merged
data/processed/data.yaml used for detector training."""
import csv
import json
import os
import random
import shutil
from pathlib import Path

import yaml

from src.data.simuletic import find_root, list_items, remap_table, source_names

P = yaml.safe_load(open("params.yaml"))["data"]
CLASSES = yaml.safe_load(open("configs/classes.yaml"))["names"]
NAME2ID = {v: int(k) for k, v in CLASSES.items()}
RAW = Path(P.get("simuletic_dir", "data/raw/simuletic"))
OUT = Path(P["processed_dir"]).parent / "simuletic"
MERGED = Path(P["processed_dir"]).parent / "data.yaml"
FRAC = P.get("simuletic_split", {"train": 0.6, "val": 0.2, "test": 0.2})


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    root = find_root(RAW)
    table = remap_table(source_names(root), CLASSES)
    items = list_items(root)

    variants = sorted({i["variant"] for i in items})
    random.Random(P.get("seed", 42)).shuffle(variants)
    n_test = max(1, round(len(variants) * FRAC["test"]))
    n_val = max(1, round(len(variants) * FRAC["val"]))
    split_of = {}
    for k, v in enumerate(variants):
        split_of[v] = "test" if k < n_test else "val" if k < n_test + n_val else "train"

    if OUT.exists():
        shutil.rmtree(OUT)
    rows, stats = [], {s: {"images": 0, "variants": set(), "baseline": {"fire": 0, "smoke": 0, "normal": 0}}
                       for s in ("train", "val", "test")}
    for it in items:
        s = split_of[it["variant"]]
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)
        lines, ids = [], set()
        for raw in it["lbl"].read_text().splitlines():
            if raw.strip():
                p = raw.split()
                new = table[int(p[0])]
                lines.append(" ".join([str(new)] + p[1:]))
                ids.add(new)
        (OUT / s / "labels" / f"{it['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        link(it["img"], OUT / s / "images" / it["img"].name)
        label = "fire" if NAME2ID["fire"] in ids else "smoke" if NAME2ID["smoke"] in ids else "normal"
        rows.append((s, f"{s}/images/{it['img'].name}", label))
        stats[s]["images"] += 1
        stats[s]["variants"].add(it["variant"])
        stats[s]["baseline"][label] += 1

    for a in stats:  # hard guarantee: no variant in more than one split
        for b in stats:
            if a < b:
                assert not (stats[a]["variants"] & stats[b]["variants"]), f"variant leak {a}/{b}"

    with open(OUT / "baseline_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image_path", "label"])
        w.writerows(rows)
    names = {int(k): v for k, v in CLASSES.items()}
    yaml.safe_dump({"train": "train/images", "val": "val/images", "test": "test/images",
                    "names": names}, open(OUT / "data.yaml", "w"), sort_keys=False)
    # merged yaml: no 'path' key, so ultralytics resolves entries relative to this file's folder
    yaml.safe_dump({"train": ["dfire/train/images", "simuletic/train/images"],
                    "val": ["dfire/val/images", "simuletic/val/images"],
                    "test": ["dfire/test/images"], "names": names},
                   open(MERGED, "w"), sort_keys=False)

    out = {s: {"images": v["images"], "variants": len(v["variants"]), "baseline": v["baseline"]}
           for s, v in stats.items()}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/simuletic_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()