"""Stage train_detector: fine-tune YOLO nano on the merged D-Fire + Simuletic data.yaml.

Usage:
  python -m src.detector.train
  python -m src.detector.train --set detector.lr0=0.002 --set detector.run_name=detector-lr2e-3 \
                               --set detector.out_dir=models/detector_lr2e-3
"""
import argparse
import json
import shutil
from pathlib import Path

from src.baseline.common import flatten, load_params
from src.tracking import mlflow_utils as mlu

EXPERIMENT = "firewatch-detector"
RUNS_DIR = Path("runs/detector").resolve()  # Ultralytics scratch output (gitignored)


def clean(d):
    """'metrics/mAP50-95(B)' -> 'val_map50_95' style names, values as floats."""
    out = {}
    for k, v in d.items():
        k = (k.replace("metrics/", "val_").replace("(B)", "").replace("val/", "val_")
              .replace("train/", "train_").replace("mAP50-95", "map50_95").replace("mAP50", "map50")
              .replace("-", "_").lower())
        out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="override, e.g. detector.lr0=0.002")
    args = ap.parse_args()
    params = load_params(args.set)
    p = params["detector"]

    from ultralytics import YOLO, settings
    # we log to MLflow ourselves (DagsHub-compatible); stop Ultralytics' own integrations
    settings.update({"mlflow": False, "dvc": False, "wandb": False, "tensorboard": False})

    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    tags = {"dataset": "dfire+simuletic", "model": Path(p["model"]).stem, "stage": "train_detector"}
    with mlu.start_run(EXPERIMENT, p["run_name"], tags) as run:
        mlu.log_params(flatten({"detector": {k: v for k, v in p.items() if k != "eval_sets"}}))

        history, best, state = [], {}, {"trained": False}

        def on_train_epoch_end(trainer):
            state["trained"] = True

        def on_fit_epoch_end(trainer):
            if not state["trained"]:
                # Ultralytics fires this once more after final_eval (validation of best.pt):
                # those are the metrics of the selected weights
                best.update(clean({k: v for k, v in trainer.metrics.items() if k.startswith("metrics/")}))
                return
            state["trained"] = False
            epoch = trainer.epoch + 1
            row = clean({**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics})
            row["lr"] = float(trainer.optimizer.param_groups[0]["lr"])
            history.append({"epoch": epoch, **row})
            mlu.log_metrics(row, step=epoch)
            fit = trainer.fitness
            if fit is not None and fit > state.get("best_fit", float("-inf")):  # same rule as best.pt
                state["best_fit"] = fit
                best.clear()
                best.update({"epoch": epoch, **row})
            print({k: round(v, 4) for k, v in row.items()})

        model = YOLO(p["model"])
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(data=str(Path(p["data"]).resolve()), imgsz=p["imgsz"], epochs=p["epochs"],
                    batch=p["batch"], lr0=p["lr0"], optimizer=p["optimizer"],
                    patience=p["patience"], fraction=p["fraction"], seed=p["seed"],
                    deterministic=True, workers=p["workers"], device=p["device"],
                    project=str(RUNS_DIR), name=p["run_name"], exist_ok=True, plots=True,
                    verbose=False)

        save_dir = Path(model.trainer.save_dir)
        shutil.copy2(save_dir / "weights" / "best.pt", out_dir / "best.pt")
        for f in ("results.csv", "results.png", "args.yaml"):
            if (save_dir / f).exists():
                shutil.copy2(save_dir / f, out_dir / f)

        meta = {"names": {int(k): v for k, v in model.names.items()}, "imgsz": p["imgsz"],
                "run_name": p["run_name"], "best_epoch": best.get("epoch"),
                "best_val_map50": best.get("val_map50"),
                "best_val_map50_95": best.get("val_map50_95"),
                "mlflow_run_id": run.info.run_id if run else None}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (Path("reports") / f"detector_train_{out_dir.name}.json").write_text(json.dumps(history, indent=2))

        if run:
            mlu.log_metrics({"best_val_map50": meta["best_val_map50"] or 0.0,
                             "best_val_map50_95": meta["best_val_map50_95"] or 0.0})
            mlu.log_artifacts([out_dir / "results.png", out_dir / "results.csv",
                               out_dir / "args.yaml", out_dir / "meta.json"])
            mlu.log_artifacts([p for p in save_dir.glob("*.png")], artifact_path="train_plots")
            mlu.log_artifacts([out_dir / "best.pt"], artifact_path="weights")
    print("best epoch", meta["best_epoch"], "val mAP50-95", meta["best_val_map50_95"])


if __name__ == "__main__":
    main()
