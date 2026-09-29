"""One-time clean and corruption evaluation of a frozen outer-fold CNN.

The script refuses to run if an evaluation-start ledger already exists. Model,
normalization, calibration, epoch count, and corruption settings are verified
against frozen SHA-256 ledgers before any outer-test image is instantiated.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pilot_fold_loader_224x224 import ROOT, SpectrogramDataset
from s2_validation_audit_224x224 import (
    corrupt,
    macro_f1,
    make_model,
    parent_aggregate,
    probability_metrics,
    softmax_numpy,
)


SEED = 2026
GROUP_COLUMN = "original-image identifier"
CLASS_COLUMN = "class ID"
CONDITIONS = [
    "clean",
    "gaussian_noise_0.05",
    "gaussian_noise_0.10",
    "time_mask_20pct",
    "zero_doppler_mask_15pct",
    "doppler_shift_8px",
    "contrast_half",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_condition_outputs(
    model, loader, device, condition: str, mean: float, std: float
):
    logits = []
    labels = []
    parents = []
    generator = torch.Generator(device=device).manual_seed(SEED)
    model.eval()
    with torch.inference_mode():
        for images, targets, parent_ids in loader:
            if tuple(images.shape[1:]) != (1, 224, 224):
                raise RuntimeError(f"Unexpected outer-test geometry: {images.shape}")
            images = images.to(device, non_blocking=True)
            if condition != "clean":
                images = corrupt(images, condition, mean, std, generator)
            outputs = model(images)
            if not torch.isfinite(outputs).all():
                raise RuntimeError(f"Nonfinite logits under {condition}")
            logits.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
            parents.append(parent_ids.numpy())
    return (
        np.concatenate(logits),
        np.concatenate(labels).astype(int),
        np.concatenate(parents).astype(int),
    )


def confusion_matrix(targets, predictions):
    matrix = np.zeros((4, 4), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


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

    output = (
        ROOT / "artifacts_224x224" / "cnn_outer_refit" /
        f"outer_{args.outer}"
    )
    checkpoint_path = output / "best.pt"
    freeze_path = output / "refit_freeze.json"
    results_path = output / "outer_test_results.json"
    marker_path = output / "outer_test_evaluation_ledger.json"
    if results_path.exists() or marker_path.exists():
        raise RuntimeError(
            "Outer-test evaluation was already started or completed for this fold; "
            "refusing duplicate evaluation."
        )

    with freeze_path.open() as file:
        freeze = json.load(file)
    if int(freeze["outer_fold"]) != args.outer:
        raise RuntimeError("Refit freeze outer-fold mismatch")
    if freeze["outer_test_dataset_instantiated"] or freeze["outer_test_images_loaded"]:
        raise RuntimeError("Refit ledger reports premature outer-test access")
    if list(freeze["frozen_test_conditions"]) != CONDITIONS:
        raise RuntimeError("Frozen corruption conditions do not match evaluator")
    if sha256(checkpoint_path) != freeze["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint hash mismatch")

    normalization_path = (
        ROOT / "artifacts_224x224" / "common" / "fold_normalization.json"
    )
    calibration_path = (
        ROOT / "artifacts_224x224" / "cnn_outer_calibration" /
        f"outer_{args.outer}" / "temperature.json"
    )
    if sha256(normalization_path) != freeze["normalization_sha256"]:
        raise RuntimeError("Normalization file changed after refit freeze")
    if sha256(calibration_path) != freeze["calibration_sha256"]:
        raise RuntimeError("Calibration file changed after refit freeze")
    with calibration_path.open() as file:
        calibration = json.load(file)
    temperature = float(calibration["deployable_temperature"])
    if not math.isclose(temperature, float(freeze["frozen_temperature"]), abs_tol=1e-12):
        raise RuntimeError("Frozen temperature mismatch")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if int(checkpoint["outer"]) != args.outer:
        raise RuntimeError("Checkpoint outer-fold mismatch")
    if str(checkpoint["selected_method"]) != str(freeze["selected_method"]):
        raise RuntimeError("Checkpoint strategy mismatch")
    if int(checkpoint["epoch"]) != int(freeze["fixed_refit_epochs"]):
        raise RuntimeError("Checkpoint epoch mismatch")
    if not math.isclose(float(checkpoint["frozen_temperature"]), temperature, abs_tol=1e-12):
        raise RuntimeError("Checkpoint temperature mismatch")

    folds = pd.read_csv(ROOT / "manifests_224x48" / "fold_manifest.csv")
    active = folds.loc[folds["outer fold"] == args.outer]
    outer_train = active.loc[active["inner fold"].notna()]
    outer_test = active.loc[active["inner fold"].isna()].reset_index(drop=True)
    train_parents = set(outer_train[GROUP_COLUMN].astype(int))
    test_parents = set(outer_test[GROUP_COLUMN].astype(int))
    if train_parents & test_parents:
        raise RuntimeError("Parent leakage between outer train and test")
    if (
        len(test_parents) != int(freeze["outer_test_parents"])
        or len(train_parents) != int(freeze["training_parents"])
    ):
        raise RuntimeError("Frozen parent counts do not match manifest")
    if outer_test["filename"].duplicated().any():
        raise RuntimeError("Duplicate outer-test image")
    if set(outer_test[CLASS_COLUMN].astype(int)) != {0, 1, 2, 3}:
        raise RuntimeError("Outer test lacks a mapped class")

    with normalization_path.open() as file:
        stats = json.load(file)["outer_refit"][f"outer_{args.outer}"]
    mean = float(stats["mean"])
    std = float(stats["standard_deviation"])
    if (
        int(stats["images"]) != len(outer_train)
        or int(stats["parents"]) != len(train_parents)
    ):
        raise RuntimeError("Outer-refit normalization ledger mismatch")
    if not math.isclose(mean, float(checkpoint["mean"]), abs_tol=1e-12):
        raise RuntimeError("Checkpoint mean mismatch")
    if not math.isclose(std, float(checkpoint["std"]), abs_tol=1e-12):
        raise RuntimeError("Checkpoint standard deviation mismatch")

    started = {
        "outer_fold": args.outer,
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "calibration_sha256": freeze["calibration_sha256"],
        "conditions": CONDITIONS,
        "rule": "one-time outer-test evaluation; no model or calibration updates",
    }
    with marker_path.open("x", encoding="utf-8") as file:
        json.dump(started, file, indent=2)
        file.write("\n")

    # Outer-test images become accessible only after every freeze assertion and
    # after the one-time evaluation-start ledger has been created.
    dataset = SpectrogramDataset(outer_test, mean, std)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    model = make_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])

    condition_reports = []
    clean_parent_ids = clean_labels = clean_logits = clean_counts = None
    clean_predictions = clean_probabilities = clean_calibrated = None
    clean_f1 = None
    for condition in CONDITIONS:
        segment_logits, segment_labels, segment_parents = collect_condition_outputs(
            model, loader, device, condition, mean, std
        )
        if len(segment_logits) != len(outer_test):
            raise RuntimeError(f"Incomplete outer-test inference for {condition}")
        aggregated = parent_aggregate(
            segment_logits, segment_labels, segment_parents
        )
        parent_ids = np.asarray([row[0] for row in aggregated], dtype=int)
        labels = np.asarray([row[1] for row in aggregated], dtype=int)
        logits = np.stack([row[2] for row in aggregated])
        counts = np.asarray([row[3] for row in aggregated], dtype=int)
        if set(parent_ids.tolist()) != test_parents:
            raise RuntimeError(f"Incomplete test-parent coverage for {condition}")

        raw_probabilities = softmax_numpy(logits)
        calibrated_probabilities = softmax_numpy(logits, temperature)
        predictions = calibrated_probabilities.argmax(axis=1)
        score, class_scores = macro_f1(labels, predictions)
        raw_metrics = probability_metrics(raw_probabilities, labels)
        calibrated_metrics = probability_metrics(calibrated_probabilities, labels)

        if condition == "clean":
            clean_parent_ids = parent_ids
            clean_labels = labels
            clean_logits = logits
            clean_counts = counts
            clean_predictions = predictions
            clean_probabilities = raw_probabilities
            clean_calibrated = calibrated_probabilities
            clean_f1 = score
        else:
            if not np.array_equal(parent_ids, clean_parent_ids):
                raise RuntimeError(f"Parent order changed under {condition}")
            if not np.array_equal(labels, clean_labels):
                raise RuntimeError(f"Labels changed under {condition}")

        condition_reports.append({
            "condition": condition,
            "parent_macro_f1": score,
            "per_class_parent_f1": class_scores,
            "delta_from_clean": 0.0 if condition == "clean" else score - clean_f1,
            "raw_ece_10_bin": raw_metrics["ece_10_bin"],
            "raw_brier": raw_metrics["brier"],
            "raw_nll": raw_metrics["nll"],
            "calibrated_ece_10_bin": calibrated_metrics["ece_10_bin"],
            "calibrated_brier": calibrated_metrics["brier"],
            "calibrated_nll": calibrated_metrics["nll"],
        })
        print(
            f"outer={args.outer} condition={condition} "
            f"parent_macro_f1={score:.4f} "
            f"delta={condition_reports[-1]['delta_from_clean']:+.4f} "
            f"calibrated_ece={calibrated_metrics['ece_10_bin']:.4f}",
            flush=True,
        )

    prediction_table = pd.DataFrame({
        "parent_measurement_id": clean_parent_ids,
        "class_id": clean_labels,
        "predicted_class_id": clean_predictions,
        "correct": clean_labels == clean_predictions,
        "segments": clean_counts,
        "raw_confidence": clean_probabilities.max(axis=1),
        "calibrated_confidence": clean_calibrated.max(axis=1),
    })
    for class_id in range(4):
        prediction_table[f"logit_{class_id}"] = clean_logits[:, class_id]
        prediction_table[f"raw_probability_{class_id}"] = clean_probabilities[:, class_id]
        prediction_table[f"calibrated_probability_{class_id}"] = clean_calibrated[:, class_id]
    prediction_table.to_csv(output / "outer_test_parent_predictions.csv", index=False)

    matrix = confusion_matrix(clean_labels, clean_predictions)
    pd.DataFrame(
        matrix,
        index=[f"actual_{index}" for index in range(4)],
        columns=[f"predicted_{index}" for index in range(4)],
    ).to_csv(output / "outer_test_confusion_matrix.csv")
    pd.DataFrame([
        {
            key: value
            for key, value in row.items()
            if key != "per_class_parent_f1"
        }
        for row in condition_reports
    ]).to_csv(output / "outer_test_robustness.csv", index=False)
    np.savez_compressed(
        output / "outer_test_clean_logits_probabilities.npz",
        parent_ids=clean_parent_ids.astype(np.int16),
        labels=clean_labels.astype(np.int8),
        logits=clean_logits,
        raw_probabilities=clean_probabilities,
        calibrated_probabilities=clean_calibrated,
    )

    report = {
        "outer_fold": args.outer,
        "selected_method": freeze["selected_method"],
        "refit_epochs": int(freeze["fixed_refit_epochs"]),
        "temperature": temperature,
        "test_images": len(outer_test),
        "test_parents": len(test_parents),
        "test_class_parent_counts": {
            str(key): int(value)
            for key, value in prediction_table["class_id"].value_counts().sort_index().items()
        },
        "clean_confusion_matrix": matrix.tolist(),
        "conditions": condition_reports,
        "protocol": {
            "split_unit": "parent measurement",
            "checkpoint_and_temperature_frozen_before_test": True,
            "outer_test_evaluations_per_fold": 1,
            "model_updates_after_test": False,
            "corruptions_frozen_before_test": CONDITIONS,
            "primary_metric": "parent-level macro-F1",
        },
    }
    temporary_results = output / "outer_test_results.json.tmp"
    temporary_results.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    temporary_results.replace(results_path)

    started["status"] = "completed"
    started["completed_utc"] = datetime.now(timezone.utc).isoformat()
    started["results_sha256"] = sha256(results_path)
    marker_path.write_text(
        json.dumps(started, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Clean outer-test parent macro-F1: {clean_f1:.4f}")
    print(f"Results: {results_path}")
    print("OUTER_TEST_224x224_EVALUATION_PASS — one-time scoring completed")


if __name__ == "__main__":
    main()
