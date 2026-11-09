"""Shared helpers for the FireWatch baseline (MobileNetV3-small, 3 classes)."""
import csv
import os
import random
from pathlib import Path

import numpy as np
import yaml

CLASS_NAMES = ["fire", "smoke", "normal"]  # index order used by the classifier only
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet stats


# ---------- params ----------
def load_params(overrides=None):
    params = yaml.safe_load(open("params.yaml"))
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = params
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)  # "0.0003" -> float, "true" -> bool
    return params


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ---------- data ----------
def read_items(csv_path, root, split, fraction=1.0, seed=42):
    """Return [(image_path, class_idx)] for one split of a baseline CSV."""
    items = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            items.append((Path(root) / row["image_path"], CLASS_TO_IDX[row["label"]]))
    if fraction < 1.0:
        random.Random(seed).shuffle(items)
        items = items[: max(1, int(len(items) * fraction))]
    return items


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(items, size, train, aug):
    import torch
    from PIL import Image
    from torchvision import transforms

    tfs = [transforms.Resize((size, size))]
    if train and aug.get("hflip"):
        tfs.append(transforms.RandomHorizontalFlip())
    cj = float(aug.get("color_jitter", 0) or 0)
    if train and cj > 0:
        tfs.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    tfs += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    tf = transforms.Compose(tfs)

    class FireDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            path, y = items[i]
            with Image.open(path) as im:
                im.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale on decode
                im = im.convert("RGB")
            return tf(im), y

    return FireDataset()


def make_loader(items, p, train):
    import torch
    ds = make_dataset(items, p["img_size"], train, p.get("augment", {}))
    return torch.utils.data.DataLoader(
        ds, batch_size=p["batch_size"], shuffle=train,
        num_workers=p["num_workers"], pin_memory=torch.cuda.is_available(),
        persistent_workers=p["num_workers"] > 0,
    )


# ---------- model ----------
def build_model(pretrained=True, freeze_backbone=False):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
    if freeze_backbone:
        for prm in model.features.parameters():
            prm.requires_grad = False
    return model


# ---------- mlflow (optional: only active when MLFLOW_TRACKING_URI is set) ----------
def mlflow_enabled():
    return bool(os.getenv("MLFLOW_TRACKING_URI"))