"""Train one compliant 224x224 CNN inner fold.

Run on an Atlas GPU. Outer-test measurements are never read by the Dataset.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from pilot_fold_loader_224x224 import ROOT, SpectrogramDataset
from pilot_balanced_sampler import BalancedBatchSampler


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 5, stride=2, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 4),
    )


def macro_f1(y_true, y_pred, classes=range(4)):
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def evaluate(model, loader, device):
    model.eval()
    loss_total = 0.0
    images_total = 0
    parent_logits = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    parent_counts = defaultdict(int)
    parent_labels = {}
    with torch.inference_mode():
        for images, labels, parents in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            images_total += batch_size
            logits_np = logits.cpu().numpy()
            labels_np = labels.cpu().numpy().tolist()
            ids = parents.tolist() if hasattr(parents, "tolist") else list(parents)
            for parent, target, row in zip(ids, labels_np, logits_np):
                parent = str(parent)
                target = int(target)
                if parent in parent_labels and parent_labels[parent] != target:
                    raise RuntimeError(f"Conflicting labels within parent {parent}")
                parent_labels[parent] = target
                parent_logits[parent] += row
                parent_counts[parent] += 1

    actual, predicted = [], []
    for parent in sorted(parent_labels):
        actual.append(parent_labels[parent])
        predicted.append(int(np.argmax(parent_logits[parent] / parent_counts[parent])))
    return {
        "validation_loss": loss_total / images_total,
        "parent_macro_f1": macro_f1(actual, predicted),
        "parents": len(parent_labels),
        "images": images_total,
        "parent_labels": actual,
        "parent_predictions": predicted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batches-per-epoch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.outer <= 5 or not 1 <= args.inner <= 5:
        parser.error("outer and inner must each be in 1..5")
    if args.epochs < 1 or args.batches_per_epoch < 1 or args.patience < 1:
        parser.error("epochs, batches-per-epoch and patience must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; run the Slurm batch on a GPU node")

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    folds = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    outer = folds.loc[folds["outer fold"] == args.outer]
    train = outer.loc[
        outer["inner fold"].notna() & (outer["inner fold"] != args.inner)
    ].reset_index(drop=True)
    validation = outer.loc[outer["inner fold"] == args.inner].reset_index(drop=True)
    outer_test = outer.loc[outer["inner fold"].isna()]
    if train.empty or validation.empty or outer_test.empty:
        raise RuntimeError("Empty train, validation, or outer-test partition")

    group_col = "original-image identifier"
    groups = [set(part[group_col]) for part in (train, validation, outer_test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Parent leakage across train, validation, outer test")
    if len(train["filename"].unique()) != len(train) or len(validation["filename"].unique()) != len(validation):
        raise RuntimeError("Duplicate images within an active partition")
    labels = set(train["class ID"]) | set(validation["class ID"])
    if labels != {0, 1, 2, 3}:
        raise RuntimeError(f"Expected four mapped classes, got {labels}")

    with (ROOT / "artifacts_224x224/common/fold_normalization.json").open() as f:
        all_stats = json.load(f)
    stats = all_stats["inner_training"][f"outer_{args.outer}_inner_{args.inner}"]
    if stats["images"] != len(train):
        raise RuntimeError("Incorrect training-fold normalization")
    mean, std = stats["mean"], stats["standard_deviation"]
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise RuntimeError("Invalid training-fold statistics")

    train_data = SpectrogramDataset(train, mean, std)
    val_data = SpectrogramDataset(validation, mean, std)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda")
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = Path(ROOT) / "artifacts_224x224" / "cnn_inner" / f"outer_{args.outer}_inner_{args.inner}"
    output.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_loss = float("inf")
    stale = 0
    history = []
    print(f"Train: {len(train)} images / {len(groups[0])} parents", flush=True)
    print(f"Validation: {len(validation)} images / {len(groups[1])} parents", flush=True)
    print(f"Outer test: {len(groups[2])} parents (never loaded)", flush=True)

    for epoch in range(1, args.epochs + 1):
        # Refresh parent/segment sampling each epoch while preserving reproducibility.
        sampler = BalancedBatchSampler(
            train, batch_size=16, batches_per_epoch=args.batches_per_epoch,
            seed=2026 + epoch - 1,
        )
        train_loader = DataLoader(
            train_data, batch_sampler=sampler, num_workers=2, pin_memory=True
        )
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for images, targets, _ in train_loader:
            if tuple(images.shape[1:]) != (1, 224, 224):
                raise RuntimeError(f"Unexpected model input shape: {images.shape}")
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * targets.size(0)
            train_count += targets.size(0)

        metrics = evaluate(model, val_loader, device)
        if metrics["images"] != len(validation) or metrics["parents"] != len(groups[1]):
            raise RuntimeError("Validation did not cover every image and parent")
        row = {
            "epoch": epoch,
            "train_loss": train_loss_total / train_count,
            "validation_loss": metrics["validation_loss"],
            "parent_macro_f1": metrics["parent_macro_f1"],
            "train_samples_drawn": train_count,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"parent_macro_f1={row['parent_macro_f1']:.4f}", flush=True
        )
        improved = (row["parent_macro_f1"] > best_score + 1e-12 or
                    (abs(row["parent_macro_f1"] - best_score) <= 1e-12 and
                     row["validation_loss"] < best_loss - 1e-12))
        if improved:
            best_score = row["parent_macro_f1"]
            best_loss = row["validation_loss"]
            stale = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "outer": args.outer, "inner": args.inner,
                "epoch": epoch, "parent_macro_f1": best_score,
                "validation_loss": best_loss, "mean": mean, "std": std,
                "class_count": 4, "seed": 2026,
                "input_geometry": [1, 224, 224],
                "source_geometry": [255, 44],
                "crop_fraction_each_side": 0.05,
                "resize": "bilinear",
            }
            temp = output / "best.pt.tmp"
            torch.save(checkpoint, temp)
            os.replace(temp, output / "best.pt")
        else:
            stale += 1
        (output / "history.json").write_text(
            json.dumps({"config": vars(args), "epochs": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    print(f"Best validation parent macro-F1: {best_score:.4f}", flush=True)
    print(f"Checkpoint: {output / 'best.pt'}", flush=True)
    print("INNER_FOLD_224x224_TRAIN_PASS — outer test untouched", flush=True)


if __name__ == "__main__":
    main()
