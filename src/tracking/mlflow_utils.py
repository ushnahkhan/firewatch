"""Small MLflow helpers shared by the detector stages.

Tracking is only active when MLFLOW_TRACKING_URI is set (plus MLFLOW_TRACKING_USERNAME /
MLFLOW_TRACKING_PASSWORD for DagsHub). Without it every helper is a no-op, so the pipeline
still runs offline.

Smoke test (creates one tiny run in 'firewatch-detector'):
  python -m src.tracking.mlflow_utils
"""
import os
from contextlib import contextmanager
from pathlib import Path


def enabled():
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


@contextmanager
def start_run(experiment, run_name=None, tags=None, run_id=None):
    """Start (or resume, with run_id) a run; yields the run, or None when tracking is off."""
    if not enabled():
        yield None
        return
    import mlflow
    if run_id is None:
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params(params):
    if enabled():
        import mlflow
        # MLflow caps param values at 6000 chars; str() keeps lists/dicts loggable
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})


def log_metrics(metrics, step=None):
    if enabled():
        import mlflow
        clean = {k.replace("(", "").replace(")", ""): float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float))}
        mlflow.log_metrics(clean, step=step)


def log_artifacts(paths, artifact_path=None):
    if enabled():
        import mlflow
        for p in paths:
            p = Path(p)
            if p.is_dir():
                mlflow.log_artifacts(str(p), artifact_path=artifact_path)
            elif p.exists():
                mlflow.log_artifact(str(p), artifact_path=artifact_path)


if __name__ == "__main__":
    if not enabled():
        raise SystemExit("set MLFLOW_TRACKING_URI / USERNAME / PASSWORD first")
    with start_run("firewatch-detector", "smoke-test", {"stage": "smoke_test"}) as run:
        log_params({"hello": "world"})
        log_metrics({"dummy": 1.0}, step=0)
        print("logged run", run.info.run_id, "to", os.getenv("MLFLOW_TRACKING_URI"))
