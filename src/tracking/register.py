"""Register the best detector in the MLflow model registry as 'firewatch-detector' (Staging).

The run is chosen by VALIDATION mAP@0.5:0.95 (meta.json written by src.detector.train),
never by a test score, so the test sets stay an unbiased estimate.

The weights are wrapped as an MLflow pyfunc model and uploaded as plain run artifacts
(registered_model/), because DagsHub's MLflow server does not support MLflow 3 logged models
(same approach as src/baseline/register.py).

Usage (MLflow env vars set):
  python -m src.tracking.register
  python -m src.tracking.register --model-dirs models/detector models/detector_lr2e-3
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from src.tracking import mlflow_utils as mlu

NAME = "firewatch-detector"
ARTIFACT = "registered_model"


class YoloDetector(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper: input = DataFrame with an 'image_path' column; output = one row per box."""

    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        import pandas as pd
        rows = []
        for path in model_input["image_path"]:
            r = self.model.predict(str(path), verbose=False)[0]
            for cls, conf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                rows.append({"image_path": path, "class": self.model.names[int(cls)],
                             "confidence": conf, "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]})
        return pd.DataFrame(rows, columns=["image_path", "class", "confidence", "x1", "y1", "x2", "y2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", default=["models/detector", "models/detector_lr2e-3"])
    args = ap.parse_args()
    if not mlu.enabled():
        sys.exit("set MLFLOW_TRACKING_URI / USERNAME / PASSWORD first")

    cands = []
    for d in args.model_dirs:
        meta_path = Path(d) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("mlflow_run_id") and meta.get("best_val_map50_95") is not None:
                cands.append((meta["best_val_map50_95"], meta["run_name"], meta["mlflow_run_id"], d))
    if not cands:
        sys.exit("no model dir with an mlflow_run_id in meta.json; train with MLflow enabled first")
    for score, name, rid, _ in sorted(cands, reverse=True):
        print(f"candidate {name}: best val mAP50-95 {score:.4f} (run {rid})")
    score, name, run_id, model_dir = max(cands)
    print(f"selected {name} by validation mAP50-95")

    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        local = str(Path(tmp) / "model")
        mlflow.pyfunc.save_model(local, python_model=YoloDetector(),
                                 artifacts={"weights": str(Path(model_dir) / "best.pt")},
                                 pip_requirements=["ultralytics", "pandas"])
        client.log_artifacts(run_id, local, artifact_path=ARTIFACT)
    print(f"uploaded model files to run {run_id} under '{ARTIFACT}/'")

    mv = mlflow.register_model(f"runs:/{run_id}/{ARTIFACT}", NAME)
    print(f"registered {NAME} version {mv.version}")
    client.update_model_version(NAME, mv.version,
                                description=f"{name}: YOLO nano, best val mAP50-95 {score:.4f}")
    try:  # classic stage (shown in the DagsHub registry UI); deprecated in MLflow 3 but supported
        client.transition_model_version_stage(NAME, mv.version, "Staging")
        print("stage -> Staging")
    except Exception as e:
        print("could not set stage Staging:", e)
    try:  # MLflow 3 replacement for stages
        client.set_registered_model_alias(NAME, "staging", mv.version)
        print("alias 'staging' set")
    except Exception as e:
        print("could not set 'staging' alias:", e)


if __name__ == "__main__":
    main()
