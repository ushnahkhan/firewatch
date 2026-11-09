import json
import sys
from pathlib import Path

import yaml
from PIL import Image

PARAMS = yaml.safe_load(open("params.yaml"))["data"]
CLASSES = yaml.safe_load(open("configs/classes.yaml"))["names"]
RAW = Path(PARAMS["raw_dir"])
DATA = RAW / "data"
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png"}
MAX_EXAMPLES = 20
HARD = ["missing_label", "orphan_label", "unreadable_image",
        "bad_line_format", "bad_class_id"]
SOFT = ["bbox_out_of_range"]


def check_class_order():
    names = yaml.safe_load(open(RAW / "data.yaml"))["names"]
    expected = [CLASSES[i] for i in sorted(CLASSES)]
    return [] if list(names) == expected else [
        f"class order mismatch: data.yaml={names} configs/classes.yaml={expected}"]


def check_split(split):
    img_dir, lbl_dir = DATA / split / "images", DATA / split / "labels"
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
    issues = {k: [] for k in HARD + SOFT}
    counts = {"images": len(imgs), "labels": len(lbls), "empty_labels": 0, "boxes": 0}
    issues["missing_label"] = sorted(set(imgs) - set(lbls))
    issues["orphan_label"] = sorted(set(lbls) - set(imgs))

    for stem, p in imgs.items():
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            issues["unreadable_image"].append(stem)

    for stem, p in lbls.items():
        lines = [l.split() for l in p.read_text().splitlines() if l.strip()]
        if not lines:
            counts["empty_labels"] += 1
        for parts in lines:
            counts["boxes"] += 1
            if len(parts) != 5:
                issues["bad_line_format"].append(stem)
                continue
            try:
                cid = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                issues["bad_line_format"].append(stem)
                continue
            if cid not in CLASSES:
                issues["bad_class_id"].append(stem)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                issues["bbox_out_of_range"].append(stem)
    return counts, issues, set(imgs)


def main():
    errors, warnings = check_class_order(), []
    report = {"classes": CLASSES, "splits": {}}
    all_stems, total_images, soft_files = {}, 0, set()

    for s in SPLITS:
        counts, issues, stems = check_split(s)
        all_stems[s] = stems
        total_images += counts["images"]
        expected = PARAMS["expected_counts"][s]
        if counts["images"] != expected:
            errors.append(f"{s}: expected {expected} images, found {counts['images']}")
        for k in HARD:
            if issues[k]:
                errors.append(f"{s}: {k} ({len(issues[k])} files)")
        for k in SOFT:
            if issues[k]:
                warnings.append(f"{s}: {k} ({len(set(issues[k]))} files, fixed in prepare)")
                soft_files |= {(s, f) for f in issues[k]}
        counts["issues"] = {k: {"count": len(v), "examples": v[:MAX_EXAMPLES]}
                            for k, v in issues.items()}
        report["splits"][s] = counts

    for a in SPLITS:
        for b in SPLITS:
            if a < b and (all_stems[a] & all_stems[b]):
                errors.append(f"filename overlap between {a} and {b}: "
                              f"{len(all_stems[a] & all_stems[b])}")

    frac = len(soft_files) / max(total_images, 1)
    report["bad_bbox_fraction"] = round(frac, 5)
    if frac > PARAMS["max_bad_bbox_fraction"]:
        errors.append(f"bad bbox fraction {frac:.4f} exceeds "
                      f"{PARAMS['max_bad_bbox_fraction']}")

    report["errors"], report["warnings"] = errors, warnings
    report["passed"] = not errors
    Path("reports").mkdir(exist_ok=True)
    Path("reports/validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "errors": errors,
                      "warnings": warnings}, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()