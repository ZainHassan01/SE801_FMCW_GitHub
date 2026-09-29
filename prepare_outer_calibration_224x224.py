"""Fit one leakage-safe temperature scaler per outer fold.

The calibration set consists only of parent-level out-of-fold logits from the
five selected inner-validation checkpoints. Outer-test images are never
instantiated by this script.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pilot_fold_loader_224x224 import ROOT, SpectrogramDataset
from s2_validation_audit_224x224 import (
    collect_clean_outputs,
    fit_temperature,
    macro_f1,
    make_model,
    parent_aggregate,
    probability_metrics,
    softmax_numpy,
)


SEED = 2026
GROUP_COLUMN = "original-image identifier"
CLASS_COLUMN = "class ID"
METHOD_DIRECTORIES = {
    "baseline": "cnn_inner",
    "latent_diversity": "cnn_latent_diversity",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_method(outer: int) -> str:
    path = (
        ROOT / "artifacts_224x224" / "cnn_inner_cv_summary" /
        "outer_fold_model_selection.csv"
    )
    table = pd.read_csv(path)
    row = table.loc[table["outer_fold"] == outer]
    if len(row) != 1:
        raise RuntimeError(f"Expected one selection row for outer fold {outer}")
    method = str(row.iloc[0]["selected_sampling_strategy"])
    if method not in METHOD_DIRECTORIES:
        raise RuntimeError(f"Unsupported selected method: {method}")
    return method


def checkpoint_path(method: str, outer: int, inner: int) -> Path:
    return (
        ROOT / "artifacts_224x224" / METHOD_DIRECTORIES[method] /
        f"outer_{outer}_inner_{inner}" / "best.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.outer <= 5:
        parser.error("outer must be in 1..5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on an Atlas GPU node")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    method = selected_method(args.outer)
    folds = pd.read_csv(ROOT / "manifests_224x48" / "fold_manifest.csv")
    active = folds.loc[folds["outer fold"] == args.outer].copy()
    outer_train = active.loc[active["inner fold"].notna()].copy()
    outer_test = active.loc[active["inner fold"].isna()].copy()
    outer_train_parents = set(outer_train[GROUP_COLUMN].astype(int))
    outer_test_parents = set(outer_test[GROUP_COLUMN].astype(int))
    if outer_train_parents & outer_test_parents:
        raise RuntimeError("Parent leakage between outer train and test")
    if set(outer_train[CLASS_COLUMN].astype(int)) != {0, 1, 2, 3}:
        raise RuntimeError("Outer training partition lacks a mapped class")

    stats_path = ROOT / "artifacts_224x224" / "common" / "fold_normalization.json"
    with stats_path.open() as file:
        all_stats = json.load(file)

    parent_rows = []
    seen_parents = set()
    checkpoint_hashes = {}
    for inner in range(1, 6):
        inner_train = outer_train.loc[outer_train["inner fold"] != inner]
        validation = outer_train.loc[
            outer_train["inner fold"] == inner
        ].reset_index(drop=True)
        train_parents = set(inner_train[GROUP_COLUMN].astype(int))
        validation_parents = set(validation[GROUP_COLUMN].astype(int))
        if train_parents & validation_parents:
            raise RuntimeError(f"Parent leakage in outer {args.outer}, inner {inner}")
        if validation_parents & seen_parents:
            raise RuntimeError("A calibration parent appears in multiple inner folds")
        if validation_parents & outer_test_parents:
            raise RuntimeError("Outer-test parent entered calibration")

        key = f"outer_{args.outer}_inner_{inner}"
        stats = all_stats["inner_training"][key]
        if (
            int(stats["images"]) != len(inner_train)
            or int(stats["parents"]) != len(train_parents)
        ):
            raise RuntimeError(f"Normalization ledger mismatch for {key}")
        mean = float(stats["mean"])
        std = float(stats["standard_deviation"])

        path = checkpoint_path(method, args.outer, inner)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (int(checkpoint["outer"]), int(checkpoint["inner"])) != (
            args.outer, inner
        ):
            raise RuntimeError(f"Checkpoint fold mismatch: {path}")
        if not math.isclose(float(checkpoint["mean"]), mean, abs_tol=1e-12):
            raise RuntimeError(f"Checkpoint mean mismatch: {path}")
        if not math.isclose(float(checkpoint["std"]), std, abs_tol=1e-12):
            raise RuntimeError(f"Checkpoint standard deviation mismatch: {path}")
        if method == "latent_diversity" and checkpoint.get("sampler") != (
            "training-only latent diversity"
        ):
            raise RuntimeError(f"Latent-diversity ledger missing: {path}")

        model = make_model().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        dataset = SpectrogramDataset(validation, mean, std)
        loader = DataLoader(
            dataset, batch_size=128, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        segment_logits, _, segment_labels, segment_parents = (
            collect_clean_outputs(model, loader, device)
        )
        if len(segment_logits) != len(validation):
            raise RuntimeError(f"Incomplete validation inference for {key}")
        aggregated = parent_aggregate(
            segment_logits, segment_labels, segment_parents
        )
        if {row[0] for row in aggregated} != validation_parents:
            raise RuntimeError(f"Incomplete parent inference for {key}")
        for parent, label, logits, count in aggregated:
            parent_rows.append({
                "outer_fold": args.outer,
                "source_inner_fold": inner,
                "parent_measurement_id": int(parent),
                "class_id": int(label),
                "segments": int(count),
                **{
                    f"logit_{index}": float(value)
                    for index, value in enumerate(logits)
                },
            })
        seen_parents.update(validation_parents)
        checkpoint_hashes[str(inner)] = sha256(path)
        print(
            f"outer={args.outer} inner={inner} calibration_parents="
            f"{len(validation_parents)} images={len(validation)}",
            flush=True,
        )
        del model, dataset, loader
        torch.cuda.empty_cache()

    if seen_parents != outer_train_parents:
        missing = sorted(outer_train_parents - seen_parents)
        extra = sorted(seen_parents - outer_train_parents)
        raise RuntimeError(f"OOF parent coverage mismatch; missing={missing}, extra={extra}")

    table = pd.DataFrame(parent_rows).sort_values("parent_measurement_id")
    if table["parent_measurement_id"].duplicated().any():
        raise RuntimeError("Duplicate OOF calibration parent")
    logits = table[[f"logit_{index}" for index in range(4)]].to_numpy(
        dtype=np.float64
    )
    labels = table["class_id"].to_numpy(dtype=int)
    raw_probabilities = softmax_numpy(logits)
    raw_f1, raw_class_f1 = macro_f1(labels, raw_probabilities.argmax(axis=1))
    raw_metrics = probability_metrics(raw_probabilities, labels)

    # Cross-fitted metrics describe calibration quality without fitting the
    # calibrator on the parent being scored. The deployable temperature below
    # is then fitted on all OOF parents and frozen before outer-test inference.
    crossfit_probabilities = np.empty_like(raw_probabilities)
    crossfit_temperatures = []
    for held_out in range(len(labels)):
        keep = np.arange(len(labels)) != held_out
        temperature = fit_temperature(logits[keep], labels[keep])
        crossfit_temperatures.append(temperature)
        crossfit_probabilities[held_out] = softmax_numpy(
            logits[held_out:held_out + 1], temperature
        )[0]
    crossfit_metrics = probability_metrics(crossfit_probabilities, labels)
    deployable_temperature = fit_temperature(logits, labels)
    calibrated_probabilities = softmax_numpy(logits, deployable_temperature)
    calibrated_metrics = probability_metrics(calibrated_probabilities, labels)

    output = (
        ROOT / "artifacts_224x224" / "cnn_outer_calibration" /
        f"outer_{args.outer}"
    )
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "oof_parent_logits.csv", index=False)
    np.savez_compressed(
        output / "oof_parent_logits_probabilities.npz",
        parent_ids=table["parent_measurement_id"].to_numpy(dtype=np.int16),
        source_inner_folds=table["source_inner_fold"].to_numpy(dtype=np.int8),
        labels=labels.astype(np.int8),
        logits=logits,
        raw_probabilities=raw_probabilities,
        crossfit_probabilities=crossfit_probabilities,
        calibrated_probabilities=calibrated_probabilities,
    )
    report = {
        "outer_fold": args.outer,
        "selected_method": method,
        "seed": SEED,
        "calibration_unit": "parent measurement",
        "oof_parents": len(table),
        "oof_images": int(table["segments"].sum()),
        "class_parent_counts": {
            str(key): int(value)
            for key, value in table["class_id"].value_counts().sort_index().items()
        },
        "uncalibrated_parent_macro_f1": raw_f1,
        "uncalibrated_per_class_parent_f1": raw_class_f1,
        "uncalibrated_metrics": raw_metrics,
        "crossfit_calibration_metrics": crossfit_metrics,
        "deployable_temperature": deployable_temperature,
        "deployable_in_sample_oof_metrics": calibrated_metrics,
        "checkpoint_sha256_by_inner_fold": checkpoint_hashes,
        "leakage_controls": {
            "logit_source": "five inner-validation partitions",
            "each_parent_predicted_by_checkpoint_excluding_that_parent": True,
            "outer_test_manifest_parents_checked_for_overlap": True,
            "outer_test_dataset_instantiated": False,
            "outer_test_images_loaded": False,
            "temperature_frozen_before_outer_refit_evaluation": True,
        },
        "corruptions_frozen_before_outer_test": [
            "clean",
            "gaussian_noise_0.05",
            "gaussian_noise_0.10",
            "time_mask_20pct",
            "zero_doppler_mask_15pct",
            "doppler_shift_8px",
            "contrast_half",
        ],
    }
    temporary = output / "temperature.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "temperature.json")

    print(f"Outer fold: {args.outer}")
    print(f"Selected method: {method}")
    print(f"OOF calibration parents: {len(table)}")
    print(f"OOF parent macro-F1: {raw_f1:.4f}")
    print(
        f"ECE raw={raw_metrics['ece_10_bin']:.4f} "
        f"crossfit={crossfit_metrics['ece_10_bin']:.4f}"
    )
    print(f"Frozen deployable temperature: {deployable_temperature:.6f}")
    print(f"Output: {output}")
    print("OUTER_CALIBRATION_224x224_PASS — outer-test images never loaded")


if __name__ == "__main__":
    main()
