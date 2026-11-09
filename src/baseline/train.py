"""Stage train_baseline: fine-tune MobileNetV3-small on D-Fire baseline labels.

Usage:
  python -m src.baseline.train
  python -m src.baseline.train --set baseline.lr=0.0003 --set baseline.run_name=lr3e-4 \
                               --set baseline.out_dir=models/baseline_lr3e-4
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.baseline.common import (CLASS_NAMES, build_model, flatten, load_params,
                                 make_loader, mlflow_enabled, read_items, set_seed)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            y_true += y.tolist()
            y_pred += out.argmax(1).tolist()
    n = len(y_true)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / n
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return total_loss / n, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="override, e.g. baseline.lr=0.0003")
    args = ap.parse_args()
    params = load_params(args.set)
    p, d = params["baseline"], params["data"]

    set_seed(p["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, csv_path = Path(d["processed_dir"]), Path(d["processed_dir"]) / "baseline_labels.csv"
    train_items = read_items(csv_path, root, "train", p["subset_fraction"], p["seed"])
    val_items = read_items(csv_path, root, "val", p["subset_fraction"], p["seed"])
    print(f"device={device} train={len(train_items)} val={len(val_items)}")
    train_loader, val_loader = make_loader(train_items, p, True), make_loader(val_items, p, False)

    model = build_model(True, p["freeze_backbone"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                  lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"])

    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    run = None
    if mlflow_enabled():
        import mlflow
        mlflow.set_experiment("firewatch-baseline")
        run = mlflow.start_run(run_name=p["run_name"])
        mlflow.set_tags({"dataset": "dfire", "model": "mobilenet_v3_small", "stage": "train_baseline"})
        mlflow.log_params(flatten(params))

    history, best_f1 = [], -1.0
    for epoch in range(1, p["epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        row = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                   val_acc=va_acc, val_f1=va_f1)
        history.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        if run:
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        if va_f1 > best_f1:  # keep the best epoch by validation macro-F1
            best_f1 = va_f1
            torch.save(model.state_dict(), out_dir / "model.pt")

    meta = {"classes": CLASS_NAMES, "img_size": p["img_size"], "best_val_f1": best_f1,
            "run_name": p["run_name"], "mlflow_run_id": run.info.run_id if run else None}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (Path("reports") / f"baseline_train_{out_dir.name}.json").write_text(json.dumps(history, indent=2))

    if run:
        import mlflow.pytorch
        model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
        mlflow.log_metric("best_val_f1", best_f1)
        mlflow.pytorch.log_model(model.cpu(), artifact_path="model")
        mlflow.end_run()
    print("best val macro-F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()