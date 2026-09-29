"""One short CNN diagnostic run using the protocol-compliant 224x224 pipeline."""

import argparse
import json

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()
    if not (1 <= args.outer <= 5 and 1 <= args.inner <= 5):
        parser.error("outer and inner must be in 1..5")

    torch.manual_seed(2026)
    np.random.seed(2026)
    rng = np.random.default_rng(2026)

    folds = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    outer = folds.loc[folds["outer fold"] == args.outer]
    train = outer.loc[
        outer["inner fold"].notna() & (outer["inner fold"] != args.inner)
    ].reset_index(drop=True)
    validation = outer.loc[outer["inner fold"] == args.inner].copy()
    outer_test = outer.loc[outer["inner fold"].isna()].copy()

    group = "original-image identifier"
    train_parents = set(train[group])
    val_parents = set(validation[group])
    test_parents = set(outer_test[group])
    if any(a & b for a, b in (
        (train_parents, val_parents),
        (train_parents, test_parents),
        (val_parents, test_parents),
    )):
        raise RuntimeError("Parent leakage across active partitions")

    # Diagnostic validation sample: at most eight images from every parent.
    selected = []
    for _, block in validation.groupby(group):
        selected.extend(
            rng.choice(
                block.index.to_numpy(), size=min(8, len(block)), replace=False
            ).tolist()
        )
    validation = validation.loc[selected].reset_index(drop=True)

    stats_path = ROOT / "artifacts_224x224/common/fold_normalization.json"
    with stats_path.open() as file:
        all_stats = json.load(file)
    stats = all_stats["inner_training"][
        f"outer_{args.outer}_inner_{args.inner}"
    ]
    if stats["images"] != len(train) or stats["parents"] != len(train_parents):
        raise RuntimeError("Incorrect active training-fold normalization")

    train_data = SpectrogramDataset(
        train, stats["mean"], stats["standard_deviation"]
    )
    val_data = SpectrogramDataset(
        validation, stats["mean"], stats["standard_deviation"]
    )
    sampler = BalancedBatchSampler(
        train, batch_size=16, batches_per_epoch=32, seed=2026
    )
    train_loader = DataLoader(
        train_data, batch_sampler=sampler, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    device = torch.device("cuda")
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    train_loss = 0.0
    train_count = 0
    for images, labels, _ in train_loader:
        if tuple(images.shape[1:]) != (1, 224, 224):
            raise RuntimeError(f"Unexpected image batch shape: {images.shape}")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)
        train_count += len(labels)

    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.inference_mode():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            val_loss += loss.item() * len(labels)
            val_count += len(labels)

    print(f"Train: {train_count} sampled images, {len(train_parents)} parents")
    print(f"Validation: {val_count} sampled images, {len(val_parents)} parents")
    print(f"Outer test: {len(test_parents)} parents (never loaded)")
    print(f"Train loss: {train_loss / train_count:.4f}")
    print(f"Validation loss: {val_loss / val_count:.4f}")
    print("CNN_224x224_SMOKE_PASS — outer test untouched")


if __name__ == "__main__":
    main()
