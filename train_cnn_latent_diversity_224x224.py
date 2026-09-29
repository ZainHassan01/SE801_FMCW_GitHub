"""Train Track-S2 CNN with a training-only latent-diversity sampler."""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from pilot_fold_loader_224x224 import ROOT, SpectrogramDataset
from pilot_balanced_sampler import BalancedBatchSampler
from train_cnn_inner import evaluate, make_model


SEED = 2026
CLASS_COUNT = 4
GROUP_COLUMN = "original-image identifier"
CLASS_COLUMN = "class ID"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_fixed_warmup(dataset, frame, device, epochs, batches_per_epoch):
    """Fixed training-only warm-up; validation is never evaluated here."""
    seed_everything(SEED)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        sampler = BalancedBatchSampler(
            frame, batch_size=16, batches_per_epoch=batches_per_epoch,
            seed=SEED + epoch - 1,
        )
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=2, pin_memory=True
        )
        model.train()
        total = count = 0
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite warm-up loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        print(
            f"warmup_epoch={epoch:02d} train_loss={total / count:.4f} "
            f"(training partition only)", flush=True,
        )
    return model


def extract_training_embeddings(model, dataset, device):
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )
    values = []
    model.eval()
    layers = list(model.children())[:-1]
    with torch.inference_mode():
        for number, (images, _, _) in enumerate(loader, start=1):
            features = images.to(device, non_blocking=True)
            for layer in layers:
                features = layer(features)
            values.append(features.cpu().numpy().astype(np.float32))
            if number % 50 == 0:
                print(f"embedding_batches={number}/{len(loader)}", flush=True)
    embeddings = np.concatenate(values)
    if len(embeddings) != len(dataset) or embeddings.shape[1] != 64:
        raise RuntimeError(f"Unexpected training embedding shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Nonfinite training embeddings")
    return embeddings


def mean_pairwise_cosine(values):
    normalized = values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12
    )
    count = len(normalized)
    if count < 2:
        return 1.0
    summed = normalized.sum(axis=0, dtype=np.float64)
    return float((summed @ summed - count) / (count * (count - 1)))


def farthest_point_indices(values, quota):
    """Deterministic cosine-distance farthest-point selection."""
    count = len(values)
    if count <= quota:
        return np.arange(count, dtype=np.int64)
    normalized = values.astype(np.float64)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    centroid = normalized.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    first = int(np.argmin(normalized @ centroid))
    selected = [first]
    minimum_distance = 1.0 - normalized @ normalized[first]
    minimum_distance[first] = -np.inf
    while len(selected) < quota:
        chosen = int(np.argmax(minimum_distance))
        selected.append(chosen)
        distance = 1.0 - normalized @ normalized[chosen]
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=int)] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def construct_training_only_pool(frame, embeddings, quota, output):
    selected_global = []
    selection_rows = []
    audit_rows = []
    for parent, indices in frame.groupby(GROUP_COLUMN, sort=True).groups.items():
        global_indices = np.asarray(list(indices), dtype=np.int64)
        parent_embeddings = embeddings[global_indices]
        local_selected = farthest_point_indices(parent_embeddings, quota)
        chosen = global_indices[local_selected]
        selected_global.extend(chosen.tolist())
        full_redundancy = mean_pairwise_cosine(parent_embeddings)
        selected_redundancy = mean_pairwise_cosine(embeddings[chosen])
        class_ids = frame.loc[global_indices, CLASS_COLUMN].unique()
        if len(class_ids) != 1:
            raise RuntimeError(f"Parent {parent} has conflicting labels")
        audit_rows.append({
            "parent_measurement_id": int(parent),
            "class_id": int(class_ids[0]),
            "available_segments": len(global_indices),
            "selected_segments": len(chosen),
            "full_mean_pairwise_cosine": full_redundancy,
            "selected_mean_pairwise_cosine": selected_redundancy,
            "cosine_redundancy_change": selected_redundancy - full_redundancy,
        })
        for rank, index in enumerate(chosen.tolist(), start=1):
            row = frame.iloc[index]
            selection_rows.append({
                "training_row_index": index,
                "selection_rank_within_parent": rank,
                "filename": row["filename"],
                "class_id": int(row[CLASS_COLUMN]),
                "parent_measurement_id": int(row[GROUP_COLUMN]),
            })

    selected = np.asarray(sorted(selected_global), dtype=np.int64)
    if len(selected) != len(set(selected.tolist())):
        raise RuntimeError("Duplicate latent-diversity selections")
    if not set(selected).issubset(set(range(len(frame)))):
        raise RuntimeError("Selection contains a non-training row")
    selection_frame = pd.DataFrame(selection_rows).sort_values(
        ["class_id", "parent_measurement_id", "selection_rank_within_parent"]
    )
    audit_frame = pd.DataFrame(audit_rows)
    selection_frame.to_csv(output / "latent_diversity_training_pool.csv", index=False)
    audit_frame.to_csv(output / "latent_diversity_redundancy.csv", index=False)
    summary = {
        "construction_partition": "active inner-training partition only",
        "validation_embeddings_used": False,
        "outer_test_embeddings_used": False,
        "distance": "cosine",
        "method": "deterministic farthest-point sampling within each parent measurement",
        "quota_per_parent": quota,
        "selected_segments": int(len(selected)),
        "training_parents": int(frame[GROUP_COLUMN].nunique()),
        "mean_full_parent_cosine": float(audit_frame["full_mean_pairwise_cosine"].mean()),
        "mean_selected_parent_cosine": float(audit_frame["selected_mean_pairwise_cosine"].mean()),
        "selected_index_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }
    (output / "latent_diversity_construction.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return selected, summary


class LatentDiversityBatchSampler(Sampler):
    """Equal classes, distinct parents, selected training segments only."""
    def __init__(self, frame, selected_indices, batch_size, batches_per_epoch, seed):
        if batch_size % CLASS_COUNT:
            raise ValueError("batch_size must be divisible by four")
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.parents_per_class = batch_size // CLASS_COUNT
        selected_frame = frame.iloc[selected_indices].copy()
        selected_frame["training_row_index"] = selected_indices
        self.lookup = {}
        for class_id in range(CLASS_COUNT):
            class_frame = selected_frame.loc[selected_frame[CLASS_COLUMN] == class_id]
            parent_lookup = {
                int(parent): block["training_row_index"].to_numpy(dtype=np.int64)
                for parent, block in class_frame.groupby(GROUP_COLUMN)
            }
            if len(parent_lookup) < self.parents_per_class:
                raise RuntimeError(
                    f"Class {class_id} has too few selected training parents"
                )
            self.lookup[class_id] = parent_lookup

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.batches_per_epoch):
            batch = []
            for class_id in range(CLASS_COUNT):
                parent_lookup = self.lookup[class_id]
                parents = np.asarray(sorted(parent_lookup), dtype=int)
                chosen_parents = rng.choice(
                    parents, size=self.parents_per_class, replace=False
                )
                for parent in chosen_parents:
                    batch.append(int(rng.choice(parent_lookup[int(parent)])))
            rng.shuffle(batch)
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=128)
    parser.add_argument("--quota-per-parent", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batches-per-epoch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    if not (1 <= args.outer <= 5 and 1 <= args.inner <= 5):
        parser.error("outer and inner must be in 1..5")
    if min(
        args.warmup_epochs, args.warmup_batches, args.quota_per_parent,
        args.epochs, args.batches_per_epoch, args.patience,
    ) < 1:
        parser.error("All training parameters must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on an Atlas GPU node")

    seed_everything(SEED)
    folds = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    outer = folds.loc[folds["outer fold"] == args.outer]
    train = outer.loc[
        outer["inner fold"].notna() & (outer["inner fold"] != args.inner)
    ].reset_index(drop=True)
    validation = outer.loc[outer["inner fold"] == args.inner].reset_index(drop=True)
    outer_test = outer.loc[outer["inner fold"].isna()].reset_index(drop=True)
    groups = [set(frame[GROUP_COLUMN]) for frame in (train, validation, outer_test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Parent leakage across train, validation, or outer test")
    if set(train[CLASS_COLUMN]) != set(range(CLASS_COUNT)):
        raise RuntimeError("Training partition does not contain all four classes")

    with (ROOT / "artifacts_224x224/common/fold_normalization.json").open() as file:
        stats = json.load(file)["inner_training"][
            f"outer_{args.outer}_inner_{args.inner}"
        ]
    if stats["images"] != len(train) or stats["parents"] != len(groups[0]):
        raise RuntimeError("Training-only normalization ledger mismatch")
    mean, std = float(stats["mean"]), float(stats["standard_deviation"])
    train_dataset = SpectrogramDataset(train, mean, std)
    device = torch.device("cuda")

    output = (
        Path(ROOT) / "artifacts_224x224" / "cnn_latent_diversity" /
        f"outer_{args.outer}_inner_{args.inner}"
    )
    output.mkdir(parents=True, exist_ok=True)
    print(f"Training-only warm-up: {len(train)} images / {len(groups[0])} parents")
    print(f"Validation held back during sampler construction: {len(groups[1])} parents")
    print(f"Outer test untouched: {len(groups[2])} parents", flush=True)

    warmup_model = train_fixed_warmup(
        train_dataset, train, device, args.warmup_epochs, args.warmup_batches
    )
    embeddings = extract_training_embeddings(warmup_model, train_dataset, device)
    selected, construction = construct_training_only_pool(
        train, embeddings, args.quota_per_parent, output
    )
    del warmup_model, embeddings
    torch.cuda.empty_cache()
    print(
        f"Latent pool frozen: {len(selected)} segments; "
        f"mean cosine {construction['mean_full_parent_cosine']:.4f} -> "
        f"{construction['mean_selected_parent_cosine']:.4f}", flush=True,
    )

    # Validation is instantiated only after the training-only selection is frozen.
    validation_dataset = SpectrogramDataset(validation, mean, std)
    validation_loader = DataLoader(
        validation_dataset, batch_size=128, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    seed_everything(SEED)  # Fresh model; same deterministic initialization as baseline.
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_score = -1.0
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        sampler = LatentDiversityBatchSampler(
            train, selected, batch_size=16,
            batches_per_epoch=args.batches_per_epoch,
            seed=SEED + epoch - 1,
        )
        loader = DataLoader(
            train_dataset, batch_sampler=sampler, num_workers=2, pin_memory=True
        )
        model.train()
        total = count = 0
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite diversity-training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)

        metrics = evaluate(model, validation_loader, device)
        if metrics["images"] != len(validation) or metrics["parents"] != len(groups[1]):
            raise RuntimeError("Incomplete inner-validation evaluation")
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "validation_loss": metrics["validation_loss"],
            "parent_macro_f1": metrics["parent_macro_f1"],
            "train_samples_drawn": count,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"parent_macro_f1={row['parent_macro_f1']:.4f}", flush=True,
        )
        improved = (
            row["parent_macro_f1"] > best_score + 1e-12 or
            (
                abs(row["parent_macro_f1"] - best_score) <= 1e-12 and
                row["validation_loss"] < best_loss - 1e-12
            )
        )
        if improved:
            best_score = row["parent_macro_f1"]
            best_loss = row["validation_loss"]
            stale = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "outer": args.outer, "inner": args.inner,
                "epoch": epoch, "parent_macro_f1": best_score,
                "validation_loss": best_loss,
                "mean": mean, "std": std, "seed": SEED,
                "class_count": CLASS_COUNT,
                "input_geometry": [1, 224, 224],
                "sampler": "training-only latent diversity",
                "quota_per_parent": args.quota_per_parent,
                "selected_segments": len(selected),
                "selection_sha256": construction["selected_index_sha256"],
            }
            temporary = output / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            os.replace(temporary, output / "best.pt")
        else:
            stale += 1
        (output / "history.json").write_text(
            json.dumps({"config": vars(args), "epochs": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    print(f"Best diversity validation parent macro-F1: {best_score:.4f}")
    print(f"Checkpoint: {output / 'best.pt'}")
    print("LATENT_DIVERSITY_224x224_PASS — selection used training only; outer test untouched")


if __name__ == "__main__":
    main()
