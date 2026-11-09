"""Register the best baseline in the MLflow model registry as 'firewatch-baseline'.

The run is chosen by VALIDATION macro-F1 (stored in each model's meta.json), never by test
score, so the test set stays an unbiased estimate.

The weights are re-saved as a standard MLflow model and uploaded as plain run artifacts
(registered_model/), because DagsHub's MLflow server does not support MLflow 3 logged models.

Usage (MLflow env vars set):
  python -m src.baseline.register
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient

from src.baseline.common import build_model

NAME = "firewatch-baseline"
ARTIFACT = "registered_model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", default=["models/baseline", "models/baseline_lr3e-4"])
    args = ap.parse_args()
    if not os.getenv("MLFLOW_TRACKING_URI"):
        sys.exit("set MLFLOW_TRACKING_URI / USERNAME / PASSWORD first")

    cands = []
    for d in args.model_dirs:
        meta_path = Path(d) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("mlflow_run_id"):
                cands.append((meta["best_val_f1"], meta["run_name"], meta["mlflow_run_id"], d))
    if not cands:
        sys.exit("no model dir with an mlflow_run_id in meta.json; train with MLflow enabled first")
    for f1, name, rid, _ in sorted(cands, reverse=True):
        print(f"candidate {name}: best val macro-F1 {f1:.4f} (run {rid})")
    f1, name, run_id, model_dir = max(cands)
    print(f"selected {name} by validation macro-F1")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu"))
    model.eval()

    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        local = str(Path(tmp) / "model")
        mlflow.pytorch.save_model(model, local, serialization_format="pickle")
        client.log_artifacts(run_id, local, artifact_path=ARTIFACT)
    print(f"uploaded model files to run {run_id} under '{ARTIFACT}/'")

    mv = mlflow.register_model(f"runs:/{run_id}/{ARTIFACT}", NAME)
    print(f"registered {NAME} version {mv.version}")
    try:  # MLflow 3 replaced stages with aliases; 'staging' alias = Staging
        client.set_registered_model_alias(NAME, "staging", mv.version)
        print("alias 'staging' set")
    except Exception as e:
        print("could not set 'staging' alias (the registry still has the model):", e)


if __name__ == "__main__":
    main()