"""Track-S2 audit for one protocol-compliant CNN inner-fold checkpoint.

Produces parent-level classification, latent-space diagnostics, leave-one-parent-
out temperature calibration, Grad-CAM images, and representation-corruption
robustness. The outer test partition is checked for group separation but is
never instantiated or evaluated.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from pilot_fold_loader_224x224 import ROOT, SpectrogramDataset


SEED = 2026
CLASS_COUNT = 4


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
        nn.Linear(64, CLASS_COUNT),
    )


def forward_with_features(model, images):
    features = images
    for layer in list(model.children())[:-1]:
        features = layer(features)
    return model[-1](features), features


def softmax_numpy(logits, temperature=1.0):
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def macro_f1(targets, predictions):
    values = []
    for class_id in range(CLASS_COUNT):
        tp = np.sum((targets == class_id) & (predictions == class_id))
        fp = np.sum((targets != class_id) & (predictions == class_id))
        fn = np.sum((targets == class_id) & (predictions != class_id))
        denominator = 2 * tp + fp + fn
        values.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(values)), values


def probability_metrics(probabilities, targets, bins=10):
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[targets]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    selected = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    nll = float(-np.mean(np.log(selected)))
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_rows = []
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(mask.sum())
        if count:
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            ece += count / len(targets) * abs(accuracy - mean_confidence)
        else:
            accuracy = mean_confidence = None
        bin_rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence,
        })
    return {"ece_10_bin": float(ece), "brier": brier, "nll": nll, "bins": bin_rows}


def fit_temperature(logits, targets):
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=100,
        tolerance_grad=1e-10, tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0))


def parent_aggregate(segment_logits, segment_labels, segment_parents):
    rows = []
    for parent in sorted(np.unique(segment_parents).tolist()):
        mask = segment_parents == parent
        labels = np.unique(segment_labels[mask])
        if len(labels) != 1:
            raise RuntimeError(f"Conflicting labels for parent {parent}")
        rows.append((int(parent), int(labels[0]), segment_logits[mask].mean(axis=0), int(mask.sum())))
    return rows


def collect_clean_outputs(model, loader, device):
    logits, features, labels, parents = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for images, target, parent in loader:
            if tuple(images.shape[1:]) != (1, 224, 224):
                raise RuntimeError(f"Unexpected validation shape {images.shape}")
            images = images.to(device, non_blocking=True)
            output, latent = forward_with_features(model, images)
            logits.append(output.cpu().numpy())
            features.append(latent.cpu().numpy())
            labels.append(target.numpy())
            parents.append(parent.numpy())
    return (
        np.concatenate(logits), np.concatenate(features),
        np.concatenate(labels).astype(int), np.concatenate(parents).astype(int),
    )


def corrupt(images, name, mean, std, generator):
    x = torch.clamp(images * std + mean, 0.0, 1.0)
    if name == "gaussian_noise_0.05":
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = torch.clamp(x + 0.05 * noise, 0.0, 1.0)
    elif name == "gaussian_noise_0.10":
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = torch.clamp(x + 0.10 * noise, 0.0, 1.0)
    elif name == "time_mask_20pct":
        width = int(round(x.shape[-1] * 0.20))
        start = (x.shape[-1] - width) // 2
        x[..., start:start + width] = mean
    elif name == "zero_doppler_mask_15pct":
        height = int(round(x.shape[-2] * 0.15))
        start = (x.shape[-2] - height) // 2
        x[..., start:start + height, :] = mean
    elif name == "doppler_shift_8px":
        x = torch.roll(x, shifts=8, dims=-2)
        x[..., :8, :] = mean
    elif name == "contrast_half":
        x = torch.clamp((x - 0.5) * 0.5 + 0.5, 0.0, 1.0)
    else:
        raise ValueError(name)
    return (x - mean) / std


def corrupted_parent_metrics(model, loader, device, name, mean, std):
    logits, labels, parents = [], [], []
    generator = torch.Generator(device=device).manual_seed(SEED)
    model.eval()
    with torch.inference_mode():
        for images, target, parent in loader:
            images = images.to(device, non_blocking=True)
            images = corrupt(images, name, mean, std, generator)
            logits.append(model(images).cpu().numpy())
            labels.append(target.numpy())
            parents.append(parent.numpy())
    logits = np.concatenate(logits)
    labels = np.concatenate(labels).astype(int)
    parents = np.concatenate(parents).astype(int)
    aggregate = parent_aggregate(logits, labels, parents)
    parent_logits = np.stack([row[2] for row in aggregate])
    parent_labels = np.asarray([row[1] for row in aggregate])
    probabilities = softmax_numpy(parent_logits)
    score, _ = macro_f1(parent_labels, probabilities.argmax(axis=1))
    calibration = probability_metrics(probabilities, parent_labels)
    return {
        "parent_macro_f1": score,
        "ece_10_bin": calibration["ece_10_bin"],
        "brier": calibration["brier"],
    }


def latent_audit(embeddings, labels, parents, parent_predictions, output):
    parent_rows = []
    parent_vectors = []
    unique_parents = sorted(np.unique(parents).tolist())
    for parent in unique_parents:
        mask = parents == parent
        values = embeddings[mask].astype(np.float64)
        target = int(np.unique(labels[mask])[0])
        vector = values.mean(axis=0)
        parent_vectors.append(vector)
        normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        count = len(values)
        if count > 1:
            summed = normalized.sum(axis=0)
            pairwise_cosine = float((summed @ summed - count) / (count * (count - 1)))
        else:
            pairwise_cosine = 1.0
        parent_rows.append({
            "parent_measurement_id": parent,
            "class_id": target,
            "segments": count,
            "mean_within_parent_cosine": pairwise_cosine,
            "predicted_class_id": int(parent_predictions[parent]),
            "correct": target == int(parent_predictions[parent]),
        })

    matrix = np.stack(parent_vectors)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vh[:2].T
    denominator = float(np.sum(singular ** 2))
    explained = ((singular[:2] ** 2) / denominator).tolist() if denominator else [0.0, 0.0]

    parent_labels = np.asarray([row["class_id"] for row in parent_rows])
    for index, row in enumerate(parent_rows):
        class_mask = parent_labels == row["class_id"]
        class_centroid = matrix[class_mask].mean(axis=0)
        row["distance_to_validation_class_centroid"] = float(
            np.linalg.norm(matrix[index] - class_centroid)
        )
        row["pc1"] = float(coordinates[index, 0])
        row["pc2"] = float(coordinates[index, 1])

    pd.DataFrame(parent_rows).to_csv(output / "latent_parent_audit.csv", index=False)
    np.savez_compressed(
        output / "validation_segment_embeddings.npz",
        embeddings=embeddings.astype(np.float32),
        labels=labels.astype(np.int16),
        parent_ids=parents.astype(np.int16),
    )
    return {
        "embedding_dimension": int(embeddings.shape[1]),
        "segments": int(len(embeddings)),
        "parents": int(len(parent_rows)),
        "pca_explained_variance_ratio_first_two": explained,
        "sampling_rule_constructed": False,
        "sampling_rule_note": (
            "This is a descriptive inner-validation audit only. Any future latent-diversity "
            "sampling rule must be constructed exclusively from the active training partition."
        ),
    }


def save_gradcam(model, dataset, validation, segment_logits, parent_predictions,
                 parent_targets, mean, std, output, device):
    probabilities = softmax_numpy(segment_logits)
    selected = []
    for class_id in range(CLASS_COUNT):
        candidates = [p for p, target in parent_targets.items() if target == class_id]
        candidates.sort(key=lambda p: (parent_predictions[p] != parent_targets[p], p))
        selected.extend(candidates[:2])
    selected.extend([p for p in parent_targets if parent_predictions[p] != parent_targets[p]])
    selected = list(dict.fromkeys(selected))

    directory = output / "gradcam"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = []
    activations = {}
    gradients = {}
    target_layer = model[6]

    def forward_hook(_module, _inputs, value):
        activations["value"] = value.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    try:
        parent_array = validation["original-image identifier"].to_numpy(dtype=int)
        for parent in selected:
            parent_class = parent_predictions[parent]
            indexes = np.flatnonzero(parent_array == parent)
            best = int(indexes[np.argmax(probabilities[indexes, parent_class])])
            image, target, _ = dataset[best]
            batch = image.unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            logits = model(batch)
            logits[0, parent_class].backward()
            weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            cam -= cam.min()
            cam /= torch.clamp(cam.max(), min=1e-12)
            heat = cam.cpu().numpy()
            base = np.clip(image[0].numpy() * std + mean, 0.0, 1.0)
            base_rgb = np.repeat(base[..., None], 3, axis=2)
            heat_rgb = np.stack([heat, heat ** 2, np.zeros_like(heat)], axis=2)
            overlay = np.clip(0.55 * base_rgb + 0.45 * heat_rgb, 0.0, 1.0)
            filename = f"parent_{parent}_true_{int(target)}_pred_{parent_class}.png"
            Image.fromarray(np.rint(overlay * 255).astype(np.uint8), "RGB").save(directory / filename)
            metadata.append({
                "parent_measurement_id": int(parent),
                "segment_row_index": best,
                "manifest_filename": str(validation.iloc[best]["filename"]),
                "true_class_id": int(target),
                "parent_predicted_class_id": int(parent_class),
                "segment_confidence_for_parent_prediction": float(probabilities[best, parent_class]),
                "correct_parent_prediction": int(target) == int(parent_class),
                "gradcam_file": f"gradcam/{filename}",
            })
    finally:
        handle_forward.remove()
        handle_backward.remove()
    pd.DataFrame(metadata).to_csv(output / "gradcam_metadata.csv", index=False)
    return {"method": "Grad-CAM", "target_layer": "third convolution (model[6])", "images": len(metadata)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()
    if not (1 <= args.outer <= 5 and 1 <= args.inner <= 5):
        parser.error("outer and inner must be in 1..5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with Slurm on an Atlas GPU node")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

    fold = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    active = fold.loc[fold["outer fold"] == args.outer]
    train = active.loc[active["inner fold"].notna() & (active["inner fold"] != args.inner)]
    validation = active.loc[active["inner fold"] == args.inner].reset_index(drop=True)
    outer_test = active.loc[active["inner fold"].isna()]
    group = "original-image identifier"
    groups = [set(frame[group]) for frame in (train, validation, outer_test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Parent leakage across train, validation, or outer test")

    checkpoint_dir = ROOT / "artifacts_224x224" / "cnn_inner" / f"outer_{args.outer}_inner_{args.inner}"
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    if (checkpoint["outer"], checkpoint["inner"]) != (args.outer, args.inner):
        raise RuntimeError("Checkpoint fold mismatch")
    stats_path = ROOT / "artifacts_224x224/common/fold_normalization.json"
    with stats_path.open() as file:
        stats = json.load(file)["inner_training"][f"outer_{args.outer}_inner_{args.inner}"]
    mean, std = float(stats["mean"]), float(stats["standard_deviation"])
    if stats["images"] != len(train) or stats["parents"] != len(groups[0]):
        raise RuntimeError("Normalization ledger does not match active training fold")
    if not math.isclose(mean, checkpoint["mean"], abs_tol=1e-12):
        raise RuntimeError("Checkpoint and fold mean mismatch")
    if not math.isclose(std, checkpoint["std"], abs_tol=1e-12):
        raise RuntimeError("Checkpoint and fold standard deviation mismatch")

    model = make_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = SpectrogramDataset(validation, mean, std)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    segment_logits, embeddings, segment_labels, segment_parents = collect_clean_outputs(model, loader, device)
    if len(segment_logits) != len(validation) or set(segment_parents) != groups[1]:
        raise RuntimeError("Incomplete validation inference")

    aggregate = parent_aggregate(segment_logits, segment_labels, segment_parents)
    parent_ids = np.asarray([row[0] for row in aggregate], dtype=int)
    parent_labels = np.asarray([row[1] for row in aggregate], dtype=int)
    parent_logits = np.stack([row[2] for row in aggregate])
    parent_counts = np.asarray([row[3] for row in aggregate], dtype=int)
    raw_probabilities = softmax_numpy(parent_logits)
    raw_predictions = raw_probabilities.argmax(axis=1)
    clean_f1, class_f1 = macro_f1(parent_labels, raw_predictions)
    if not math.isclose(clean_f1, checkpoint["parent_macro_f1"], abs_tol=1e-10):
        raise RuntimeError("Clean audit F1 does not reproduce checkpoint F1")
    clean_calibration = probability_metrics(raw_probabilities, parent_labels)

    crossfit_probabilities = np.empty_like(raw_probabilities)
    crossfit_temperatures = []
    for held_out in range(len(parent_ids)):
        keep = np.arange(len(parent_ids)) != held_out
        temperature = fit_temperature(parent_logits[keep], parent_labels[keep])
        crossfit_temperatures.append(temperature)
        crossfit_probabilities[held_out] = softmax_numpy(
            parent_logits[held_out:held_out + 1], temperature
        )[0]
    deployable_temperature = fit_temperature(parent_logits, parent_labels)
    crossfit_calibration = probability_metrics(crossfit_probabilities, parent_labels)

    output = checkpoint_dir / "s2_validation_audit"
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame({
        "parent_measurement_id": parent_ids,
        "class_id": parent_labels,
        "predicted_class_id": raw_predictions,
        "correct": parent_labels == raw_predictions,
        "validation_segments": parent_counts,
        "raw_confidence": raw_probabilities.max(axis=1),
        "crossfit_calibrated_confidence": crossfit_probabilities.max(axis=1),
        "crossfit_temperature": crossfit_temperatures,
    })
    prediction_frame.to_csv(output / "parent_predictions_and_calibration.csv", index=False)
    np.savez_compressed(
        output / "parent_logits_probabilities.npz",
        parent_ids=parent_ids, labels=parent_labels, logits=parent_logits,
        raw_probabilities=raw_probabilities,
        crossfit_probabilities=crossfit_probabilities,
    )
    (output / "temperature.json").write_text(json.dumps({
        "deployable_temperature_fit_on_all_inner_validation_parents": deployable_temperature,
        "evaluation_method": "leave-one-parent-out cross-fitted temperature scaling",
        "crossfit_temperatures": crossfit_temperatures,
        "warning": "Do not tune or refit this temperature on the outer-test fold.",
    }, indent=2) + "\n", encoding="utf-8")

    parent_prediction_map = dict(zip(parent_ids.tolist(), raw_predictions.tolist()))
    parent_target_map = dict(zip(parent_ids.tolist(), parent_labels.tolist()))
    latent = latent_audit(
        embeddings, segment_labels, segment_parents, parent_prediction_map, output
    )
    gradcam = save_gradcam(
        model, dataset, validation, segment_logits, parent_prediction_map,
        parent_target_map, mean, std, output, device,
    )

    robustness = [{
        "corruption": "clean",
        "parent_macro_f1": clean_f1,
        "delta_from_clean": 0.0,
        "ece_10_bin": clean_calibration["ece_10_bin"],
        "brier": clean_calibration["brier"],
    }]
    corruptions = [
        "gaussian_noise_0.05", "gaussian_noise_0.10", "time_mask_20pct",
        "zero_doppler_mask_15pct", "doppler_shift_8px", "contrast_half",
    ]
    for name in corruptions:
        metrics = corrupted_parent_metrics(model, loader, device, name, mean, std)
        metrics["corruption"] = name
        metrics["delta_from_clean"] = metrics["parent_macro_f1"] - clean_f1
        robustness.append(metrics)
        print(
            f"{name}: parent_macro_f1={metrics['parent_macro_f1']:.4f} "
            f"delta={metrics['delta_from_clean']:+.4f}", flush=True,
        )
    pd.DataFrame(robustness).to_csv(output / "robustness.csv", index=False)

    report = {
        "outer_fold": args.outer,
        "inner_fold": args.inner,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_images": len(validation),
        "validation_parents": len(parent_ids),
        "outer_test_parents_not_loaded": len(groups[2]),
        "clean_parent_macro_f1": clean_f1,
        "per_class_parent_f1": class_f1,
        "uncalibrated": clean_calibration,
        "crossfit_temperature_calibrated": crossfit_calibration,
        "deployable_temperature": deployable_temperature,
        "latent_audit": latent,
        "saliency": gradcam,
        "robustness": robustness,
        "leakage_controls": {
            "split_unit": "parent measurement",
            "outer_test_evaluated": False,
            "latent_sampling_rule_constructed": False,
            "calibration_crossfit_unit": "parent measurement",
            "normalization_source": "active inner-training fold only",
        },
    }
    (output / "s2_validation_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Clean parent macro-F1: {clean_f1:.4f}")
    print(
        f"Calibration ECE raw={clean_calibration['ece_10_bin']:.4f} "
        f"crossfit={crossfit_calibration['ece_10_bin']:.4f}"
    )
    print(f"Grad-CAM images: {gradcam['images']}")
    print(f"Audit directory: {output}")
    print("S2_VALIDATION_AUDIT_224x224_PASS — outer test untouched")


if __name__ == "__main__":
    main()
