"""Stage compare: baseline (MobileNetV3-small) vs detector (YOLO nano) on every shared test set.

Reads reports/baseline_metrics.json and reports/detector_metrics.json and writes
reports/comparison.md (tables) and reports/comparison.png (F1 per model and test set).

Image-level precision / recall / F1 use the same labels for both models (fire if any fire box,
else smoke, else normal) and the same macro rule (average over classes present in the set).
mAP is detector-only. Inference times are per image, batch size 1, on CPU, but measured on
different machines, so treat them as rough.
"""
import json
from pathlib import Path

REPORTS = Path("reports")


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main():
    base = json.loads((REPORTS / "baseline_metrics.json").read_text())
    det = json.loads((REPORTS / "detector_metrics.json").read_text())
    sets = [s for s in base if s in det]

    lines = ["# Baseline vs detector", "",
             "Image-level metrics use the same fire / smoke / normal labels for both models "
             "(macro average over the classes present in each test set). mAP is box-level and "
             "only exists for the detector. Inference is ms per image, batch size 1, CPU.", ""]
    rows = []
    for s in sets:
        b, d = base[s], det[s]
        lines += [f"## {s} ({b['n_images']} images)", "",
                  "| Model | Accuracy | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) |",
                  "|---|---|---|---|---|---|---|---|",
                  f"| Baseline (MobileNetV3-small) | {fmt(b['accuracy'])} | {fmt(b['precision'])} | "
                  f"{fmt(b['recall'])} | {fmt(b['f1'])} | n/a | n/a | {fmt(b['inference_ms'], 1)} |",
                  f"| Detector (YOLO nano), image-level | {fmt(d['image']['accuracy'])} | "
                  f"{fmt(d['image']['precision'])} | {fmt(d['image']['recall'])} | {fmt(d['image']['f1'])} | "
                  f"{fmt(d['box']['map50'])} | {fmt(d['box']['map50_95'])} | {fmt(d['inference_ms'], 1)} |",
                  "",
                  f"Detector box-level: precision {fmt(d['box']['precision'])}, recall "
                  f"{fmt(d['box']['recall'])}, F1 {fmt(d['box']['f1'])}. Per class mAP@0.5: " +
                  ", ".join(f"{c} {fmt(v['map50'])}" for c, v in d["box"]["per_class"].items()) + ".",
                  ""]
        rows.append((s, b["f1"], d["image"]["f1"]))
    (REPORTS / "comparison.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = range(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.6))
    b1 = ax.bar([i - w / 2 for i in x], [r[1] for r in rows], w, label="Baseline (MobileNetV3-small)",
                color="#4C78A8")
    b2 = ax.bar([i + w / 2 for i in x], [r[2] for r in rows], w, label="Detector (YOLO nano)",
                color="#F58518")
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(list(x), [r[0] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("image-level macro F1")
    ax.set_title("Baseline vs detector")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "comparison.png", dpi=150)
    plt.close(fig)
    print((REPORTS / "comparison.md").read_text())


if __name__ == "__main__":
    main()
