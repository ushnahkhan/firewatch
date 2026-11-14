"""Stage evaluate_detector: box metrics + image-level metrics on every test set in
params.yaml (detector.eval_sets), written to reports/detector_metrics.json.

Box metrics (Ultralytics val, split=test): precision, recall, f1, mAP@0.5, mAP@0.5:0.95,
overall and per class.
Image-level metrics: each image gets fire / smoke / normal from its predicted boxes with the
same rule as the baseline labels (fire if any fire box, else smoke, else normal), so the
detector can be compared with the MobileNetV3 baseline on equal terms.

Usage:
  python -m src.detector.evaluate
  python -m src.detector.evaluate --set detector.out_dir=models/detector_lr2e-3
"""
import argparse
import json
import time
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from src.baseline.common import CLASS_NAMES, load_params, read_items
from src.baseline.evaluate import plot_cm
from src.tracking import mlflow_utils as mlu

RUNS_DIR = Path("runs/detector_eval").resolve()
N_TIMING = 100  # images used for the batch-1 CPU timing (same as the baseline)


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def box_metrics(model, es, p, tag):
    res = model.val(data=str(Path(es["data"]).resolve()), split=es["split"], imgsz=p["imgsz"],
                    batch=p["batch"], device=p["device"], project=str(RUNS_DIR), name=tag,
                    exist_ok=True, plots=True, verbose=False)
    b = res.box
    per_class = {}
    for i, c in enumerate(res.ap_class_index):
        cp, cr, ap50, ap = b.class_result(i)
        per_class[model.names[int(c)]] = {"precision": float(cp), "recall": float(cr),
                                          "f1": f1(cp, cr), "map50": float(ap50),
                                          "map50_95": float(ap)}
    return {"precision": float(b.mp), "recall": float(b.mr), "f1": f1(b.mp, b.mr),
            "map50": float(b.map50), "map50_95": float(b.map), "per_class": per_class,
            "val_speed_ms": {k: float(v) for k, v in res.speed.items()}}, Path(res.save_dir)


def image_level(model, es, p):
    items = read_items(Path(es["root"]) / "baseline_labels.csv", es["root"], es["split"])
    y_true, y_pred = [], []
    for path, y in items:
        r = model.predict(str(path), conf=p["conf"], imgsz=p["imgsz"], device=p["device"],
                          verbose=False)[0]
        found = {model.names[int(c)] for c in r.boxes.cls.tolist()}
        label = "fire" if "fire" in found else "smoke" if "smoke" in found else "normal"
        y_true.append(y)
        y_pred.append(CLASS_NAMES.index(label))
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    present = [c for c in CLASS_NAMES if rep[c]["support"] > 0]  # same macro rule as baseline

    def macro(key):
        return sum(rep[c][key] for c in present) / len(present)

    return {"n_images": len(items), "accuracy": rep["accuracy"], "precision": macro("precision"),
            "recall": macro("recall"), "f1": macro("f1-score"), "macro_over_classes": present,
            "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")}
                          for c in CLASS_NAMES},
            "confusion_matrix": cm.tolist()}, items, cm


def inference_ms_cpu(weights, items, p):
    """Mean model time per image on CPU, batch size 1 (preprocess + forward + NMS)."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    paths = [str(x) for x, _ in items[:N_TIMING]]
    for x in paths[:5]:  # warm-up
        m.predict(x, imgsz=p["imgsz"], device="cpu", verbose=False)
    times = []
    for x in paths:
        t = time.perf_counter()
        m.predict(x, imgsz=p["imgsz"], device="cpu", verbose=False)
        times.append((time.perf_counter() - t) * 1000)
    return sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    p = load_params(args.set)["detector"]

    from ultralytics import YOLO, settings
    settings.update({"mlflow": False, "dvc": False, "wandb": False, "tensorboard": False})

    out_dir = Path(p["out_dir"])
    weights = out_dir / "best.pt"
    model = YOLO(str(weights))
    Path("reports").mkdir(exist_ok=True)

    plot_dir = Path("reports") / f"detector_eval_{out_dir.name}"  # one folder of plots (a DVC out)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results, artifacts = {}, [plot_dir]
    for es in p["eval_sets"]:
        if not Path(es["data"]).exists():
            print(f"skip {es['name']}: {es['data']} not found")
            continue
        tag = f"{out_dir.name}_{es['name']}"
        box, val_dir = box_metrics(model, es, p, tag)
        img, items, cm = image_level(model, es, p)
        ms = inference_ms_cpu(weights, items, p)
        results[es["name"]] = {"box": box, "image": img, "inference_ms": ms}

        plot_cm(cm, f"detector (image-level) - {es['name']}", plot_dir / f"image_cm_{es['name']}.png")
        for name in ("PR_curve.png", "BoxPR_curve.png", "confusion_matrix.png"):
            if (val_dir / name).exists():  # Ultralytics' box-level plots
                (plot_dir / f"{es['name']}_{name}").write_bytes((val_dir / name).read_bytes())
        print(es["name"], "box:", {k: round(box[k], 4) for k in ("precision", "recall", "map50", "map50_95")},
              "image:", {k: round(img[k], 4) for k in ("precision", "recall", "f1")},
              "cpu ms:", round(ms, 1))

    out = Path("reports") / ("detector_metrics.json" if out_dir.name == "detector"
                             else f"detector_metrics_{out_dir.name}.json")
    out.write_text(json.dumps(results, indent=2))

    meta = json.loads((out_dir / "meta.json").read_text())
    if mlu.enabled() and meta.get("mlflow_run_id"):
        with mlu.start_run("firewatch-detector", run_id=meta["mlflow_run_id"]):  # resume train run
            for name, r in results.items():
                mlu.log_metrics({f"{name}_{k}": r["box"][k]
                                 for k in ("precision", "recall", "f1", "map50", "map50_95")})
                mlu.log_metrics({f"{name}_img_{k}": r["image"][k]
                                 for k in ("accuracy", "precision", "recall", "f1")})
                mlu.log_metrics({f"{name}_inference_ms": r["inference_ms"]})
            mlu.log_artifacts(artifacts, artifact_path="eval")
            mlu.log_artifacts([out])


if __name__ == "__main__":
    main()
