"""Stage evaluate_baseline: metrics on every test set listed in params.yaml (baseline.eval_sets).

Usage:
  python -m src.baseline.evaluate
  python -m src.baseline.evaluate --set baseline.out_dir=models/baseline_lr3e-4
"""
import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.baseline.common import (CLASS_NAMES, build_model, load_params, make_dataset,
                                 mlflow_enabled, read_items)


@torch.no_grad()
def predict(model, items, p, device):
    loader = torch.utils.data.DataLoader(make_dataset(items, p["img_size"], False, {}),
                                         batch_size=p["batch_size"], num_workers=p["num_workers"])
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += model(x.to(device)).argmax(1).tolist()
        y_true += y.tolist()
    return y_true, y_pred


@torch.no_grad()
def inference_ms(model, items, p, device, n=100):
    """Mean forward-pass time, batch size 1, image decoding excluded."""
    ds = make_dataset(items[:n], p["img_size"], False, {})
    x0 = ds[0][0].unsqueeze(0).to(device)
    for _ in range(5):  # warm-up
        model(x0)
    times = []
    for i in range(len(ds)):
        x = ds[i][0].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return sum(times) / len(times)


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    params = load_params(args.set)
    p = params["baseline"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(p["out_dir"])

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    model.to(device).eval()

    results, figs = {}, []
    Path("reports").mkdir(exist_ok=True)
    for es in p["eval_sets"]:
        csv_path = Path(es["root"]) / "baseline_labels.csv"
        if not csv_path.exists():
            print(f"skip {es['name']}: {csv_path} not found")
            continue
        items = read_items(csv_path, es["root"], es["split"])
        y_true, y_pred = predict(model, items, p, device)
        rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        present = [c for c in CLASS_NAMES if rep[c]["support"] > 0]  # skip classes absent from this set

        def macro(key):
            return sum(rep[c][key] for c in present) / len(present)

        results[es["name"]] = {
            "n_images": len(items),
            "accuracy": rep["accuracy"],
            "precision": macro("precision"),
            "recall": macro("recall"),
            "f1": macro("f1-score"),
            "macro_over_classes": present,
            "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")}
                          for c in CLASS_NAMES},
            "confusion_matrix": cm.tolist(),
            "inference_ms": inference_ms(model, items, p, device),
        }
        fig = Path("reports") / f"baseline_cm_{es['name']}_{out_dir.name}.png"
        plot_cm(cm, f"baseline - {es['name']}", fig)
        figs.append(fig)
        print(es["name"], {k: round(v, 4) for k, v in results[es["name"]].items()
                           if isinstance(v, float)})

    out = Path("reports") / ("baseline_metrics.json" if out_dir.name == "baseline"
                             else f"baseline_metrics_{out_dir.name}.json")
    out.write_text(json.dumps(results, indent=2))

    meta = json.loads((out_dir / "meta.json").read_text())
    if mlflow_enabled() and meta.get("mlflow_run_id"):
        import mlflow
        with mlflow.start_run(run_id=meta["mlflow_run_id"]):  # resume the training run
            for name, r in results.items():
                mlflow.log_metrics({f"{name}_{k}": r[k] for k in
                                    ("accuracy", "precision", "recall", "f1", "inference_ms")})
            for fig in figs:
                mlflow.log_artifact(str(fig))
            mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()