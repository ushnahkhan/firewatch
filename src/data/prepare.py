import csv
import json
import os
import shutil
from pathlib import Path

import yaml

P = yaml.safe_load(open("params.yaml"))["data"]
CLASSES = yaml.safe_load(open("configs/classes.yaml"))["names"]
NAME2ID = {v: k for k, v in CLASSES.items()}
RAW = Path(P["raw_dir"]) / "data"
OUT = Path(P["processed_dir"])
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png"}
TOL = 1e-6


def clean_box(parts):
    """Return (cleaned_line_or_None, status) where status is ok/clipped/dropped."""
    cid = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    if w <= 0 or h <= 0:
        return None, "dropped"
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    c = [min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2)]
    if max(abs(a - b) for a, b in zip(c, (x1, y1, x2, y2))) < TOL:
        return " ".join(parts), "ok"
    nw, nh = c[2] - c[0], c[3] - c[1]
    if nw <= 0 or nh <= 0:
        return None, "dropped"
    return (f"{cid} {(c[0] + c[2]) / 2:.6f} {(c[1] + c[3]) / 2:.6f} "
            f"{nw:.6f} {nh:.6f}"), "clipped"


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def baseline_label(ids):
    if NAME2ID["fire"] in ids:
        return "fire"
    if NAME2ID["smoke"] in ids:
        return "smoke"
    return "normal"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    stats, rows = {}, []
    for s in SPLITS:
        img_in, lbl_in = RAW / s / "images", RAW / s / "labels"
        img_out, lbl_out = OUT / s / "images", OUT / s / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        st = {"images": 0, "boxes_in": 0, "boxes_out": 0, "dropped": 0,
              "clipped": 0, "baseline": {"fire": 0, "smoke": 0, "normal": 0}}
        for img in sorted(p for p in img_in.iterdir() if p.suffix.lower() in IMG_EXT):
            lines, ids = [], set()
            for raw in (lbl_in / f"{img.stem}.txt").read_text().splitlines():
                if not raw.strip():
                    continue
                st["boxes_in"] += 1
                line, status = clean_box(raw.split())
                if status == "dropped":
                    st["dropped"] += 1
                    continue
                if status == "clipped":
                    st["clipped"] += 1
                lines.append(line)
                ids.add(int(line.split()[0]))
            (lbl_out / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            link(img, img_out / img.name)
            label = baseline_label(ids)
            st["baseline"][label] += 1
            st["images"] += 1
            st["boxes_out"] += len(lines)
            rows.append((s, f"{s}/images/{img.name}", label))
        stats[s] = st

    with open(OUT / "baseline_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image_path", "label"])
        w.writerows(rows)

        yaml.safe_dump({"train": "train/images",
                    "val": "val/images", "test": "test/images",
                    "names": {int(k): v for k, v in CLASSES.items()}},
                   open(OUT / "data.yaml", "w"), sort_keys=False)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()