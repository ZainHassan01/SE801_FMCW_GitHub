import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from pilot_fold_loader import ROOT, SpectrogramDataset
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

    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)

    folds = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    outer = folds[folds["outer fold"] == args.outer]

    train = outer[
        outer["inner fold"].notna()
        & (outer["inner fold"] != args.inner)
    ].reset_index(drop=True)

    validation = outer[
        outer["inner fold"] == args.inner
    ]

    train_parents = set(train["original-image identifier"])
    val_parents = set(validation["original-image identifier"])
    if train_parents & val_parents:
        raise RuntimeError("Parent overlap")

    # Small, reproducible validation sample: up to 8 images per parent.
    selected = []
    for _, block in validation.groupby("original-image identifier"):
        selected.extend(
            rng.choice(
                block.index.to_numpy(),
                size=min(8, len(block)),
                replace=False,
            ).tolist()
        )
    validation = validation.loc[selected].reset_index(drop=True)

    with (ROOT / "artifacts_224x48/common/fold_normalization.json").open() as f:
        all_stats = json.load(f)

    stats = all_stats["inner_training"][
        f"outer_{args.outer}_inner_{args.inner}"
    ]
    if stats["images"] != len(train):
        raise RuntimeError("Incorrect training-fold normalization")

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
        val_data, batch_size=32, shuffle=False, num_workers=2,
        pin_memory=True
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

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")

            val_loss += loss.item() * len(labels)
            val_count += len(labels)

    print(f"Train: {train_count} sampled images, {len(train_parents)} parents")
    print(f"Validation: {val_count} images, {len(val_parents)} parents")
    print(f"Train loss: {train_loss / train_count:.4f}")
    print(f"Validation loss: {val_loss / val_count:.4f}")
    print("CNN_SMOKE_PASS — diagnostic run; outer test untouched")


if __name__ == "__main__":
    main()