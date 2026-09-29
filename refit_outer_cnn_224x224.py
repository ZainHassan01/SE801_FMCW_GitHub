"""Refit one frozen Track-S2 CNN on the complete outer-training partition.

This script does not instantiate or read the outer-test image dataset. It uses
the sampling strategy and epoch count selected from the five inner folds, plus
the previously frozen OOF temperature scaler.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pilot_balanced_sampler import BalancedBatchSampler
from pilot_fold_loader_224x224 import ROOT, SpectrogramDataset
from train_cnn_inner import make_model
from train_cnn_latent_diversity_224x224 import (
    LatentDiversityBatchSampler,
    construct_training_only_pool,
    extract_training_embeddings,
    seed_everything,
    train_fixed_warmup,
)


SEED = 2026
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 256
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 4
WARMUP_BATCHES = 128
QUOTA_PER_PARENT = 32
GROUP_COLUMN = "original-image identifier"
CLASS_COLUMN = "class ID"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection(outer: int) -> dict:
    path = (
        ROOT / "artifacts_224x224" / "cnn_inner_cv_summary" /
        "outer_fold_model_selection.csv"
    )
    table = pd.read_csv(path)
    row = table.loc[table["outer_fold"] == outer]
    if len(row) != 1:
        raise RuntimeError(f"Expected one selection row for outer fold {outer}")
    result = row.iloc[0].to_dict()
    method = str(result["selected_sampling_strategy"])
    if method not in {"baseline", "latent_diversity"}:
        raise RuntimeError(f"Unsupported selected strategy: {method}")
    epochs = int(result["selected_refit_epochs"])
    if epochs < 1:
        raise RuntimeError("Invalid selected refit epoch count")
    result["selected_sampling_strategy"] = method
    result["selected_refit_epochs"] = epochs
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.outer <= 5:
        parser.error("outer must be in 1..5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on an Atlas GPU node")

    output = (
        ROOT / "artifacts_224x224" / "cnn_outer_refit" /
        f"outer_{args.outer}"
    )
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "refit_freeze.json"
    if freeze_path.exists():
        raise RuntimeError(
            f"Outer refit is already frozen: {freeze_path}. Refusing overwrite."
        )

    selection = load_selection(args.outer)
    method = selection["selected_sampling_strategy"]
    epochs = selection["selected_refit_epochs"]

    calibration_path = (
        ROOT / "artifacts_224x224" / "cnn_outer_calibration" /
        f"outer_{args.outer}" / "temperature.json"
    )
    with calibration_path.open() as file:
        calibration = json.load(file)
    if int(calibration["outer_fold"]) != args.outer:
        raise RuntimeError("Calibration outer-fold mismatch")
    if str(calibration["selected_method"]) != method:
        raise RuntimeError("Calibration method does not match model selection")
    temperature = float(calibration["deployable_temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Invalid frozen temperature")
    leakage = calibration["leakage_controls"]
    if leakage["outer_test_dataset_instantiated"] or leakage["outer_test_images_loaded"]:
        raise RuntimeError("Calibration ledger reports outer-test access")

    folds = pd.read_csv(ROOT / "manifests_224x48" / "fold_manifest.csv")
    active = folds.loc[folds["outer fold"] == args.outer]
    outer_train = active.loc[active["inner fold"].notna()].reset_index(drop=True)
    outer_test = active.loc[active["inner fold"].isna()]
    train_parents = set(outer_train[GROUP_COLUMN].astype(int))
    test_parents = set(outer_test[GROUP_COLUMN].astype(int))
    if not outer_train.empty and not outer_test.empty:
        if train_parents & test_parents:
            raise RuntimeError("Parent leakage between outer train and test")
    else:
        raise RuntimeError("Empty outer train or test manifest partition")
    if set(outer_train[CLASS_COLUMN].astype(int)) != {0, 1, 2, 3}:
        raise RuntimeError("Outer training partition lacks a mapped class")
    if outer_train["filename"].duplicated().any():
        raise RuntimeError("Duplicate outer-training image")

    normalization_path = (
        ROOT / "artifacts_224x224" / "common" / "fold_normalization.json"
    )
    with normalization_path.open() as file:
        stats = json.load(file)["outer_refit"][f"outer_{args.outer}"]
    if (
        int(stats["images"]) != len(outer_train)
        or int(stats["parents"]) != len(train_parents)
    ):
        raise RuntimeError("Outer-refit normalization ledger mismatch")
    mean = float(stats["mean"])
    std = float(stats["standard_deviation"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise RuntimeError("Invalid outer-refit normalization")

    seed_everything(SEED)
    device = torch.device("cuda")
    dataset = SpectrogramDataset(outer_train, mean, std)
    selected_indices = None
    construction = None
    if method == "latent_diversity":
        print(
            f"Outer {args.outer}: constructing latent pool from outer-training "
            f"only ({len(outer_train)} images / {len(train_parents)} parents)",
            flush=True,
        )
        warmup_model = train_fixed_warmup(
            dataset, outer_train, device, WARMUP_EPOCHS, WARMUP_BATCHES
        )
        embeddings = extract_training_embeddings(warmup_model, dataset, device)
        selected_indices, construction = construct_training_only_pool(
            outer_train, embeddings, QUOTA_PER_PARENT, output
        )
        if construction["validation_embeddings_used"]:
            raise RuntimeError("Validation embeddings entered outer refit pool")
        if construction["outer_test_embeddings_used"]:
            raise RuntimeError("Outer-test embeddings entered outer refit pool")
        del warmup_model, embeddings
        torch.cuda.empty_cache()
        print(
            f"Frozen latent pool: {len(selected_indices)} segments; cosine "
            f"{construction['mean_full_parent_cosine']:.4f} -> "
            f"{construction['mean_selected_parent_cosine']:.4f}",
            flush=True,
        )

    seed_everything(SEED)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )
    history = []
    print(f"Selected method: {method}", flush=True)
    print(f"Fixed refit epochs: {epochs}", flush=True)
    print(f"Outer-test parents recorded but images not loaded: {len(test_parents)}", flush=True)

    for epoch in range(1, epochs + 1):
        if method == "latent_diversity":
            sampler = LatentDiversityBatchSampler(
                outer_train,
                selected_indices,
                batch_size=BATCH_SIZE,
                batches_per_epoch=BATCHES_PER_EPOCH,
                seed=SEED + epoch - 1,
            )
        else:
            sampler = BalancedBatchSampler(
                outer_train,
                batch_size=BATCH_SIZE,
                batches_per_epoch=BATCHES_PER_EPOCH,
                seed=SEED + epoch - 1,
            )
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=2, pin_memory=True
        )
        model.train()
        total = 0.0
        count = 0
        for images, targets, _parents in loader:
            if tuple(images.shape[1:]) != (1, 224, 224):
                raise RuntimeError(f"Unexpected input geometry: {images.shape}")
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite outer-refit loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "samples_drawn": count,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}/{epochs:02d} train_loss={row['train_loss']:.4f}",
            flush=True,
        )

    history_path = output / "refit_history.json"
    history_path.write_text(
        json.dumps({
            "outer_fold": args.outer,
            "selected_method": method,
            "fixed_epochs": epochs,
            "epochs": history,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "outer": args.outer,
        "epoch": epochs,
        "selected_method": method,
        "mean": mean,
        "std": std,
        "class_count": 4,
        "seed": SEED,
        "input_geometry": [1, 224, 224],
        "outer_training_images": len(outer_train),
        "outer_training_parents": len(train_parents),
        "outer_test_parents_not_loaded": len(test_parents),
        "batches_per_epoch": BATCHES_PER_EPOCH,
        "learning_rate": LEARNING_RATE,
        "frozen_temperature": temperature,
        "selection_sha256": (
            construction["selected_index_sha256"]
            if construction is not None else None
        ),
    }
    temporary_checkpoint = output / "best.pt.tmp"
    final_checkpoint = output / "best.pt"
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, final_checkpoint)

    freeze = {
        "outer_fold": args.outer,
        "selected_method": method,
        "fixed_refit_epochs": epochs,
        "epoch_rule": str(selection["epoch_rule"]),
        "seed": SEED,
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256(final_checkpoint),
        "normalization_sha256": sha256(normalization_path),
        "calibration_sha256": sha256(calibration_path),
        "frozen_temperature": temperature,
        "training_images": len(outer_train),
        "training_parents": len(train_parents),
        "outer_test_parents": len(test_parents),
        "outer_test_dataset_instantiated": False,
        "outer_test_images_loaded": False,
        "training_only_latent_pool": method == "latent_diversity",
        "latent_pool_summary": construction,
        "frozen_test_conditions": calibration["corruptions_frozen_before_outer_test"],
    }
    temporary_freeze = output / "refit_freeze.json.tmp"
    temporary_freeze.write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    temporary_freeze.replace(freeze_path)

    print(f"Frozen checkpoint: {final_checkpoint}")
    print(f"Frozen temperature: {temperature:.6f}")
    print("OUTER_REFIT_224x224_PASS — outer-test images never loaded")


if __name__ == "__main__":
    main()
