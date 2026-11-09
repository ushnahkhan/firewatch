# FireWatch

FireWatch detects **fire and smoke** in camera images. Deliverable 1 builds the core ML pipeline end to end and makes it fully reproducible:

- a **baseline** image classifier (MobileNetV3-small: fire / smoke / normal),
- a **main model**, a YOLO-nano object detector that draws boxes around fire and smoke,
- data versioning with **DVC**, experiment tracking and a model registry with **MLflow**, both hosted on **DagsHub**.

Not in this deliverable: Docker, cloud deployment, REST APIs, monitoring.

- GitHub: https://github.com/ushnahkhan/firewatch
- DagsHub (data + MLflow): https://dagshub.com/ushnahkhan/firewatch
- MLflow tracking URI: `https://dagshub.com/ushnahkhan/firewatch.mlflow`

**Team:** Ushnah (repo, data, DVC), Ayesha (baseline model), Lamia (detector, MLflow, integration).

## Datasets

| Dataset | Source | Use |
|---|---|---|
| D-Fire | Kaggle `sayedgamal99/smoke-fire-detection-yolo` (about 21.5k images, YOLO boxes, own train/val/test split) | training and main test set |
| Simuletic CCTV | Kaggle `simuletic/cctv-smoke-and-fire-emergency-detection-dataset` (239 usable images, 48 scene variants) | separate domain test; also a small part of detector train/val |

FIRESENSE (Zenodo) is **not used** in this deliverable: it only has video-level labels, no bounding boxes, so a detector cannot be scored on it.

**Class map** (single source of truth: `configs/classes.yaml`): `0 = smoke`, `1 = fire`. This is D-Fire's own order. Simuletic labels are remapped **by class name** in `prepare_simuletic`, so its source order never matters.

## Folder structure

```
firewatch/
  README.md  .gitignore  dvc.yaml  dvc.lock  params.yaml  requirements.txt
  configs/classes.yaml          class map
  data/                         raw/ and processed/  (DVC-tracked, not in Git)
  src/data/                     ingest.py, validate.py, validate_stage.py, prepare.py,
                                simuletic.py, validate_simuletic.py, prepare_simuletic.py
  src/baseline/                 common.py, train.py, evaluate.py, register.py
  src/detector/                 train.py, evaluate.py        (detector, Lamia)
  src/tracking/                 MLflow helpers               (Lamia)
  reports/                      metrics, validation reports, plots
  models/                       weights (DVC-tracked, not in Git)
```

## Setup

```bash
git clone https://github.com/ushnahkhan/firewatch.git
cd firewatch
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

**Credentials are never committed.** Use these on your own machine only:

1. **Kaggle:** create a token (Kaggle > Settings > Create New Token) and save `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
2. **DagsHub token:** DagsHub > your avatar > Settings > Tokens. You must be a collaborator on the repo.
3. **DVC remote (already in `.dvc/config`); add your credentials locally:**
   ```bash
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user <your-dagshub-username>
   dvc remote modify origin --local password <your-dagshub-token>
   ```
   `--local` writes to `.dvc/config.local`, which Git ignores.
4. **MLflow** (set in each terminal that trains or evaluates):
   ```bash
   export MLFLOW_TRACKING_URI=https://dagshub.com/ushnahkhan/firewatch.mlflow
   export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
   export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
   ```
   On Windows PowerShell use `$env:MLFLOW_TRACKING_URI="..."` and the same for the other two.

## How to run

**Data pipeline (reproduce from scratch):**

```bash
dvc repro
```

This runs `ingest`, `validate`, `validate_simuletic`, `prepare`, `prepare_simuletic`, then the baseline stages. `ingest` downloads D-Fire (about 2.8 GB) and Simuletic (about 330 MB) from Kaggle **only if they are missing**. Inspect the pipeline with `dvc status` and `dvc dag`.

**Getting the data without Kaggle:** `dvc pull` works, but the dataset is about 86,000 small files and DagsHub's free HTTP remote is slow for that (hours). Prefer `dvc repro`. If a pull or push shows 502 or timeout errors, rerun it with fewer parallel jobs: `dvc pull -j 4`.

**Baseline** (from the repo root, with the MLflow variables set):

```bash
python -m src.baseline.train
python -m src.baseline.evaluate
# second configuration for comparison:
python -m src.baseline.train --set baseline.lr=0.0003 --set baseline.run_name=lr3e-4 --set baseline.out_dir=models/baseline_lr3e-4
python -m src.baseline.evaluate --set baseline.out_dir=models/baseline_lr3e-4
# register the best run (chosen by validation F1) in the MLflow registry:
python -m src.baseline.register
```

Hyperparameters (image size, batch size, epochs, learning rate, seed, augmentation) live in `params.yaml` under `baseline:`. Use `baseline.subset_fraction` (for example `0.1`) to develop on a subset; run `1.0` for the full data. Use `baseline.num_workers=0` if data loading errors on your machine.

**Detector:** see `src/detector/` (Lamia). Train on `data/processed/data.yaml`; evaluate Simuletic separately with `data/processed/simuletic/data.yaml`.

## Data validation

`validate` (D-Fire) and `validate_simuletic` fail the pipeline, with a JSON report in `reports/`, on: missing or orphan label files, unreadable images, label lines without 5 fields, invalid class ids, boxes outside 0 to 1 or with non-positive size, class-order mismatch against `configs/classes.yaml`, filename overlap between splits, and (Simuletic) a scene variant appearing in more than one split. Out-of-range D-Fire boxes are a warning below a small threshold (`max_bad_bbox_fraction`) and are cleaned in `prepare`.

What `prepare` does to D-Fire: 18 zero-size boxes are dropped and 379 boxes that overhang the image edge are clipped to the image. It also writes `baseline_labels.csv`: an image is `fire` if it has any fire box, else `smoke` if it has any smoke box, else `normal`.

One Simuletic image (`fire_detected_var16_img3`) has no label file; it is excluded via `data.simuletic_exclude` in `params.yaml` and listed in `reports/validation_simuletic.json`.

## Results

| Model | Test set | Precision | Recall | F1 | mAP@0.5 | Inference (ms) |
|---|---|---|---|---|---|---|
| Baseline (MobileNetV3-small) | D-Fire test | 0.775 | 0.798 | 0.773 | n/a | ~25 |
| Baseline (MobileNetV3-small) | Simuletic test | 0.831 | 0.817 | 0.824 | n/a | ~26 |
| Detector (YOLO nano) | D-Fire test | | | | | |
| Detector (YOLO nano) | Simuletic test | | | | | |

Baseline rows are the registered run `baseline-v1` (macro averages; Simuletic macro covers fire and smoke only because it has no `normal` images). Detector rows are filled in from `reports/detector_metrics.json`.

## Known limits

- Simuletic is small and synthetic: its test split is 50 images (30 fire, 20 smoke, no normal), so scores there are rough indications only.
- Fog vs smoke confusion and night scenes are known weak spots and are not addressed here.
- The baseline was trained on a 10% subset for 2 epochs on CPU to fit the time available; the settings are recorded in `params.yaml` and in each MLflow run. The two baseline configurations are within noise of each other.
- FIRESENSE was dropped (no bounding boxes).

## Git workflow

Branches `feat/data`, `feat/baseline`, `feat/detector`; changes go through pull requests reviewed by another member. No raw data, model weights, `.env` or tokens in Git (see `.gitignore`).