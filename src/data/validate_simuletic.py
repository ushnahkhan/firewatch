"""Stage validate_simuletic: checks the raw Simuletic data, writes
reports/validation_simuletic.json and exits non-zero on any error."""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from src.data.simuletic import EXCLUDE, find_root, list_items, remap_table, source_names

P = yaml.safe_load(open("params.yaml"))["data"]
CLASSES = yaml.safe_load(open("configs/classes.yaml"))["names"]
MIN_SIDE = 64
MAX_EXAMPLES = 20


def main():
    root = find_root(P.get("simuletic_dir", "data/raw/simuletic"))
    names = source_names(root)
    errors = []
    try:
        table = remap_table(names, CLASSES)
    except SystemExit as e:
        table = {}
        errors.append(str(e))
    items = list_items(root)
    if not items:
        errors.append("no images found")

    issues = {k: [] for k in ["missing_label", "orphan_label", "unreadable_image", "small_image",
                              "no_variant_in_name", "bad_line_format", "bad_class_id",
                              "bbox_out_of_range"]}
    class_counts, sizes, boxes = Counter(), Counter(), 0
    stems = Counter(i["stem"] for i in items)
    dup = [s for s, c in stems.items() if c > 1]
    if dup:
        errors.append(f"duplicate filenames: {dup[:MAX_EXAMPLES]}")

    for it in items:
        if it["variant"] is None:
            issues["no_variant_in_name"].append(it["stem"])
        try:
            with Image.open(it["img"]) as im:
                im.verify()
            with Image.open(it["img"]) as im:
                sizes[im.size] += 1
                if min(im.size) < MIN_SIDE:
                    issues["small_image"].append(it["stem"])
        except Exception:
            issues["unreadable_image"].append(it["stem"])
        if not it["lbl"].exists():
            issues["missing_label"].append(it["stem"])
            continue
        for line in it["lbl"].read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            boxes += 1
            if len(parts) != 5:
                issues["bad_line_format"].append(it["stem"])
                continue
            try:
                cid = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                issues["bad_line_format"].append(it["stem"])
                continue
            if cid not in names:
                issues["bad_class_id"].append(it["stem"])
            else:
                class_counts[CLASSES[table[cid]] if cid in table else str(cid)] += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                issues["bbox_out_of_range"].append(it["stem"])

    img_stems = {i["stem"] for i in items}
    lbl_dirs = {i["lbl"].parent for i in items}
    for d in lbl_dirs:
        for p in d.glob("*.txt"):
            if p.stem not in img_stems:
                issues["orphan_label"].append(p.stem)

    for k, v in issues.items():
        if v:
            errors.append(f"{k} ({len(set(v))} files)")

    report = {"root": str(root), "source_class_names": names, "remap": table,
              "excluded": sorted(EXCLUDE), "images": len(items), "boxes": boxes,
              "variants": len({i["variant"] for i in items}),
              "class_counts": dict(class_counts),
              "image_sizes": {f"{w}x{h}": c for (w, h), c in sizes.most_common(5)},
              "issues": {k: {"count": len(set(v)), "examples": sorted(set(v))[:MAX_EXAMPLES]}
                         for k, v in issues.items()},
              "errors": errors, "passed": not errors}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/validation_simuletic.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "errors": errors, "images": len(items),
                      "variants": report["variants"], "class_counts": dict(class_counts),
                      "source_class_names": names}, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()