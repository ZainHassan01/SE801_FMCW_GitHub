"""Descriptive S2 follow-up: 15-bin ECE, training embeddings, GPU profile.

Uses frozen predictions and checkpoints. Never tunes a model or re-runs test
inference. Intended to run in the existing project on the user's PC/Atlas.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 2026
CLASS_NAMES = {0: "Drone", 1: "Bird", 2: "Human", 3: "Corner reflector"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def get_root(value):
    if value is not None:
        return Path(value).expanduser().resolve()
    if os.environ.get("SE801_FMCW_ROOT"):
        return Path(os.environ["SE801_FMCW_ROOT"]).resolve()
    atlas = Path("/data/szain/SE801_FMCW")
    return atlas if atlas.is_dir() else Path(__file__).resolve().parent


def output_directory(root):
    result = root / "artifacts_224x224" / "cnn_posthoc_audit"
    result.mkdir(parents=True, exist_ok=True)
    return result


def ece_rows(probs, labels, n_bins):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[1] != 4 or probs.shape[0] != len(labels):
        raise ValueError("Expected N by 4 parent-level probabilities")
    if len(labels) == 0 or not np.isfinite(probs).all():
        raise ValueError("Empty or nonfinite probability array")
    if not np.all(np.isin(labels, list(CLASS_NAMES))):
        raise ValueError("Unexpected class labels")
    if (probs < -1e-9).any() or (probs > 1 + 1e-9).any():
        raise ValueError("Probabilities out of [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Parent probabilities must sum to one")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    # Bin i: [i/B,(i+1)/B), with confidence 1 included in final bin.
    membership = np.minimum(np.floor(confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    total = 0.0
    for index in range(n_bins):
        mask = membership == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        average_confidence = float(confidence[mask].mean()) if count else None
        component = (
            count / len(labels) * abs(accuracy - average_confidence)
            if count else 0.0
        )
        total += component
        rows.append({
            "bin": index + 1,
            "lower_inclusive": index / n_bins,
            "upper_exclusive_except_last": (index + 1) / n_bins,
            "parent_count": count,
            "accuracy": accuracy,
            "mean_confidence": average_confidence,
            "weighted_absolute_gap": component,
        })
    assert sum(row["parent_count"] for row in rows) == len(labels)
    return float(total), rows


def calibration(root):
    source = (root / "artifacts_224x224" / "cnn_final_nested_cv_summary"
              / "pooled_clean_parent_predictions.csv")
    frame = pd.read_csv(source)
    required = {"outer_fold", "parent_measurement_id", "class_id",
                "predicted_class_id"}
    raw_names = [f"raw_probability_{i}" for i in range(4)]
    scaled_names = [f"calibrated_probability_{i}" for i in range(4)]
    required.update(raw_names + scaled_names)
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {required - set(frame)}")
    if (len(frame) != 130 or frame["parent_measurement_id"].duplicated().any()
            or set(frame["outer_fold"].astype(int)) != set(range(1, 6))
            or set(frame["class_id"].astype(int)) != set(CLASS_NAMES)):
        raise ValueError("Expected 130 distinct held-out parents, five folds, four classes")
    labels = frame["class_id"].to_numpy(dtype=int)
    prediction = frame[scaled_names].to_numpy().argmax(axis=1)
    if not np.array_equal(prediction, frame["predicted_class_id"].to_numpy(dtype=int)):
        raise ValueError("Saved predictions differ from saved calibrated probabilities")
    results = {"source_sha256": digest(source), "unit": "parent measurement",
               "n_parents": len(frame), "method": "15 equal-width bins, confidence 1 in last bin",
               "model_changes": False, "outer_test_inference": False,
               "pooled": {}, "outer_folds": {}}
    bins = []
    for name, columns in (("raw", raw_names), ("temperature_scaled", scaled_names)):
        probabilities = frame[columns].to_numpy(dtype=np.float64)
        value15, rows = ece_rows(probabilities, labels, 15)
        value10, _ = ece_rows(probabilities, labels, 10)
        results["pooled"][name] = {"ece_15_bin": value15, "ece_10_bin_check": value10}
        bins.extend({"probabilities": name, **row} for row in rows)
        for fold in range(1, 6):
            subset = frame["outer_fold"].to_numpy(dtype=int) == fold
            fold_value, _ = ece_rows(probabilities[subset], labels[subset], 15)
            results["outer_folds"].setdefault(str(fold), {})[name] = {
                "parents": int(subset.sum()), "ece_15_bin": fold_value,
            }

    # Sanity-check against the previously frozen reported 10-bin numbers.
    summary_path = source.with_name("s2_final_nested_cv_summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        old = summary["pooled_out_of_fold_clean_result"]
        for name, key in (("raw", "raw_calibration"),
                          ("temperature_scaled", "temperature_scaled_calibration")):
            expected = float(old[key]["ece_10_bin"])
            actual = results["pooled"][name]["ece_10_bin_check"]
            if abs(actual - expected) > 1e-6:
                raise ValueError(f"{name} 10-bin ECE fails prior-summary consistency check")
        results["prior_summary_sha256"] = digest(summary_path)

    folder = output_directory(root)
    (folder / "ece_15_bin.json").write_text(json.dumps(results, indent=2) + "\n",
                                             encoding="utf-8")
    pd.DataFrame(bins).to_csv(folder / "ece_15_bin_details.csv", index=False)
    print("Pooled parent ECE (15 bins): raw={:.4f}, calibrated={:.4f}".format(
        results["pooled"]["raw"]["ece_15_bin"],
        results["pooled"]["temperature_scaled"]["ece_15_bin"]))
    print(f"Results: {folder / 'ece_15_bin.json'}")


def frozen_model(root, outer):
    import torch
    from s2_validation_audit_224x224 import make_model

    folder = root / "artifacts_224x224" / "cnn_outer_refit" / f"outer_{outer}"
    ckpt_path = folder / "best.pt"
    freeze = json.loads((folder / "refit_freeze.json").read_text(encoding="utf-8"))
    if freeze["checkpoint_sha256"] != digest(ckpt_path):
        raise ValueError("Frozen checkpoint SHA-256 mismatch")
    if int(freeze["outer_fold"]) != outer:
        raise ValueError("Wrong outer-fold freeze ledger")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if int(checkpoint["outer"]) != outer or int(checkpoint["class_count"]) != 4:
        raise ValueError("Wrong outer-fold checkpoint or class count")
    if (abs(float(checkpoint["mean"]) - float(freeze.get("mean", checkpoint["mean"])))
            > 1e-10):
        raise ValueError("Checkpoint normalization inconsistency")
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint, freeze


def get_training_sample(root, outer, per_parent):
    from pilot_fold_loader_224x224 import SpectrogramDataset

    frame = pd.read_csv(root / "manifests_224x48" / "fold_manifest.csv")
    active = frame.loc[frame["outer fold"] == outer]
    train = active.loc[active["inner fold"].notna()]
    test = active.loc[active["inner fold"].isna()]
    group = "original-image identifier"
    train_parents = set(train[group].astype(int))
    test_parents = set(test[group].astype(int))
    if train_parents & test_parents or not train_parents or not test_parents:
        raise RuntimeError("Outer fold parents overlap or are missing")
    rng = np.random.default_rng(SEED + outer)
    selected = []
    for _, block in train.groupby(group, sort=True):
        selected.extend(rng.choice(block.index.to_numpy(),
                                   size=min(len(block), per_parent),
                                   replace=False).tolist())
    rows = train.loc[selected].sort_values([group, "filename"]).reset_index(drop=True)
    if set(rows[group].astype(int)) != train_parents:
        raise RuntimeError("Training parent missing from embedding sample")
    ledger_path = root / "artifacts_224x224" / "common" / "fold_normalization.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    stats = ledger["outer_refit"][f"outer_{outer}"]
    if int(stats["images"]) != len(train) or int(stats["parents"]) != len(train_parents):
        raise ValueError("Statistics do not match this outer-training partition")
    dataset = SpectrogramDataset(rows, stats["mean"], stats["standard_deviation"])
    return rows, dataset, len(test_parents)


def embedding_plot(root, outer, method, per_parent):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from s2_validation_audit_224x224 import forward_with_features

    if not torch.cuda.is_available():
        raise RuntimeError("Run embedding extraction on an Atlas GPU node")
    # The existing loader resolves ROOT at import; keep it tied to this project.
    os.environ["SE801_FMCW_ROOT"] = str(root)
    rows, dataset, heldout_count = get_training_sample(root, outer, per_parent)
    model, checkpoint, freeze = frozen_model(root, outer)
    model = model.cuda().eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=True)
    chunks = []
    with torch.inference_mode():
        for images, _, _ in loader:
            _, features = forward_with_features(model, images.cuda(non_blocking=True))
            chunks.append(features.cpu().numpy())
    embeddings = np.concatenate(chunks)
    if embeddings.shape != (len(rows), 64) or not np.isfinite(embeddings).all():
        raise RuntimeError("Invalid final-refit 64-dimensional embeddings")

    algorithm = method
    if algorithm in ("auto", "umap"):
        try:
            import umap
            reduction = umap.UMAP(n_components=2, n_neighbors=15,
                                  min_dist=0.1, metric="cosine",
                                  random_state=SEED)
            algorithm = "umap"
        except ImportError:
            if algorithm == "umap":
                raise RuntimeError("UMAP unavailable: install umap-learn or use --method tsne")
            algorithm = "tsne"
    if algorithm == "tsne":
        from sklearn.manifold import TSNE
        reduction = TSNE(n_components=2, perplexity=min(30, (len(rows) - 1) // 3),
                         init="pca", learning_rate="auto", random_state=SEED)
    coordinates = reduction.fit_transform(embeddings)
    if coordinates.shape != (len(rows), 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("Invalid reduced coordinates")
    labels = rows["class ID"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    for class_id, name in CLASS_NAMES.items():
        mask = labels == class_id
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1],
                   s=12, alpha=0.62, label=f"{name} ({int(mask.sum())})")
    ax.set_xlabel(f"{algorithm.upper()} dimension 1")
    ax.set_ylabel(f"{algorithm.upper()} dimension 2")
    ax.set_title(f"S2 frozen CNN; outer {outer} training parents only")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    folder = output_directory(root)
    image_path = folder / f"outer_{outer}_training_embeddings_{algorithm}.png"
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    coordinate_table = rows[["filename", "class ID", "original-image identifier"]].copy()
    coordinate_table["axis_1"] = coordinates[:, 0]
    coordinate_table["axis_2"] = coordinates[:, 1]
    coordinate_table.to_csv(folder / f"outer_{outer}_training_embedding_coordinates.csv",
                            index=False)
    report = {
        "outer_fold": outer, "algorithm": algorithm, "seed": SEED,
        "training_only": True, "outer_test_images_loaded": False,
        "used_for_sampler_construction": False, "selected_strategy": freeze["selected_method"],
        "training_parents": int(rows["original-image identifier"].nunique()),
        "heldout_parents_excluded": heldout_count,
        "sampled_training_segments": len(rows), "quota_per_training_parent": per_parent,
        "embedding_dimension": embeddings.shape[1], "checkpoint_sha256": freeze["checkpoint_sha256"],
        "interpretation": "Descriptive training-only visualization, not independent test evidence",
    }
    (folder / f"outer_{outer}_training_embedding_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{algorithm.upper()}: {len(rows)} training segments, {report['training_parents']} parents")
    print(f"Plot: {image_path}")


def profile(root, outer, repeats, warmup):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Run profiling on an Atlas GPU node")
    model, checkpoint, freeze = frozen_model(root, outer)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    model = model.to(device).eval()
    parameters = sum(t.numel() for t in model.parameters() if t.requires_grad)
    report = {
        "outer_fold": outer, "checkpoint_sha256": freeze["checkpoint_sha256"],
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "parameters": parameters, "seed": SEED, "warmup": warmup,
        "repeats": repeats, "image_shape": [1, 224, 224],
        "timing_scope": "CUDA forward only; excludes PNG I/O, preprocessing, H2D and parent aggregation",
        "inputs": "synthetic standardized floats; no outer-test access",
        "batches": [],
    }
    for batch_size in (1, 16):
        x = torch.randn(batch_size, 1, 224, 224, device=device)
        torch.cuda.synchronize()
        for _ in range(warmup):
            with torch.inference_mode():
                model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = int(torch.cuda.memory_allocated())
        # CUDA events measure execution time on the device, not Python overhead.
        begin = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        with torch.inference_mode():
            for first, last in zip(begin, end):
                first.record()
                model(x)
                last.record()
        torch.cuda.synchronize()
        values = np.asarray([a.elapsed_time(b) for a, b in zip(begin, end)])
        report["batches"].append({
            "batch_size": batch_size, "median_batch_ms": float(np.median(values)),
            "p95_batch_ms": float(np.percentile(values, 95)),
            "mean_batch_ms": float(values.mean()),
            "mean_per_image_ms": float(values.mean() / batch_size),
            "baseline_allocated_mib": before / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        })

    # Additional single-batch diagnostic: gradient norm on actual outer-training
    # images. This is a snapshot, not a substitute for training-curve logs.
    os.environ["SE801_FMCW_ROOT"] = str(root)
    rows, dataset, heldout_count = get_training_sample(root, outer, per_parent=1)
    selected = (rows.groupby("class ID", sort=True).head(4).index.tolist())
    if len(selected) != 16:
        raise RuntimeError("Need four independent training parents per class")
    x_cpu = torch.stack([dataset[i][0] for i in selected])
    labels_cpu = torch.tensor([dataset[i][1] for i in selected], dtype=torch.long)
    x_train = x_cpu.to(device)
    labels_train = labels_cpu.to(device)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    train_baseline = torch.cuda.memory_allocated()
    loss = torch.nn.functional.cross_entropy(model(x_train), labels_train)
    loss.backward()
    torch.cuda.synchronize()
    norm = sum(float(t.grad.detach().square().sum()) for t in model.parameters()
               if t.grad is not None) ** 0.5
    report["gradient_snapshot"] = {
        "source": "one balanced batch of 16 outer-training images (four parents per class)",
        "mode": "frozen eval-mode batch norm; no optimizer step",
        "outer_test_images_loaded": False, "heldout_parents_excluded": heldout_count,
        "cross_entropy": float(loss.detach()), "gradient_l2": norm,
        "before_allocated_mib": train_baseline / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    folder = output_directory(root)
    path = folder / f"outer_{outer}_gpu_profile.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Profile: {path} | forward median (batch=1): {report['batches'][0]['median_batch_ms']:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Local or Atlas project root")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("calibration", help="Post hoc 15-bin ECE from saved parent probabilities")
    embedding = sub.add_parser("embeddings", help="Frozen CNN on sampled outer-training images")
    embedding.add_argument("--outer", type=int, choices=range(1, 6), required=True)
    embedding.add_argument("--per-parent", type=int, default=12)
    embedding.add_argument("--method", choices=("auto", "umap", "tsne"), default="auto")
    benchmark = sub.add_parser("profile", help="Frozen CNN GPU forward and training snapshot")
    benchmark.add_argument("--outer", type=int, choices=range(1, 6), required=True)
    benchmark.add_argument("--warmup", type=int, default=30)
    benchmark.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    root = get_root(args.root)
    if not root.is_dir():
        parser.error(f"Project root does not exist: {root}")
    if args.action == "calibration":
        calibration(root)
    elif args.action == "embeddings":
        if args.per_parent < 1:
            parser.error("--per-parent must be positive")
        embedding_plot(root, args.outer, args.method, args.per_parent)
    elif args.action == "profile":
        if args.warmup < 1 or args.repeats < 10:
            parser.error("Use at least 1 warmup and 10 repeats")
        profile(root, args.outer, args.repeats, args.warmup)


if __name__ == "__main__":
    main()
