"""Inspect parent-level validation predictions for a saved CNN inner fold."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pilot_fold_loader_224x224 import ROOT, SpectrogramDataset
from train_cnn_inner import evaluate, make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()
    if not (1 <= args.outer <= 5 and 1 <= args.inner <= 5):
        parser.error("outer and inner must be in 1..5")

    output = Path(ROOT) / "artifacts_224x224" / "cnn_inner" / f"outer_{args.outer}_inner_{args.inner}"
    checkpoint_path = output / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["outer"], checkpoint["inner"]) != (args.outer, args.inner):
        raise RuntimeError("Checkpoint fold does not match requested fold")

    fold = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    active = fold.loc[fold["outer fold"] == args.outer]
    train = active.loc[active["inner fold"].notna() & (active["inner fold"] != args.inner)]
    val = active.loc[active["inner fold"] == args.inner].reset_index(drop=True)
    test = active.loc[active["inner fold"].isna()]
    group = "original-image identifier"
    if val.empty or train.empty or test.empty:
        raise RuntimeError("Missing fold partition")
    if set(val[group]) & (set(train[group]) | set(test[group])):
        raise RuntimeError("Parent overlap across partitions")

    stats_path = ROOT / "artifacts_224x224/common/fold_normalization.json"
    with stats_path.open() as f:
        stats = json.load(f)["inner_training"][f"outer_{args.outer}_inner_{args.inner}"]
    if stats["images"] != len(train):
        raise RuntimeError("Normalization did not come from the active training fold")
    if not np.isclose(checkpoint["mean"], stats["mean"], rtol=0, atol=1e-12):
        raise RuntimeError("Checkpoint mean and fold mean differ")
    if not np.isclose(checkpoint["std"], stats["standard_deviation"], rtol=0, atol=1e-12):
        raise RuntimeError("Checkpoint std and fold std differ")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on Atlas GPU node")
    model = make_model().cuda()
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(
        SpectrogramDataset(val, stats["mean"], stats["standard_deviation"]),
        batch_size=128, shuffle=False, num_workers=2, pin_memory=True,
    )
    metrics = evaluate(model, loader, torch.device("cuda"))

    ids = sorted(str(p) for p in val[group].unique())
    if metrics["images"] != len(val) or metrics["parents"] != len(ids):
        raise RuntimeError("Incomplete validation predictions")
    parent_info = {}
    for p, rows in val.groupby(group):
        classes = rows["class ID"].unique()
        if len(classes) != 1:
            raise RuntimeError(f"Parent {p} has conflicting labels")
        parent_info[str(p)] = (int(classes[0]), len(rows))
    if set(ids) != set(parent_info):
        raise RuntimeError("Validation parent IDs do not match")

    predictions = []
    confusion = np.zeros((4, 4), dtype=int)
    for parent, actual, predicted in zip(ids, metrics["parent_labels"], metrics["parent_predictions"]):
        label, count = parent_info[parent]
        if actual != label or not (0 <= actual < 4 and 0 <= predicted < 4):
            raise RuntimeError("Prediction order or class ID mismatch")
        confusion[actual, predicted] += 1
        predictions.append({
            "parent_measurement_id": parent,
            "class_id": actual,
            "predicted_class_id": predicted,
            "correct": actual == predicted,
            "validation_images": count,
        })

    summary = []
    for c in range(4):
        tp = int(confusion[c, c])
        support = int(confusion[c].sum())
        predicted_count = int(confusion[:, c].sum())
        denom = support + predicted_count
        summary.append({
            "class_id": c,
            "parents": support,
            "recall": tp / support if support else 0.0,
            "precision": tp / predicted_count if predicted_count else 0.0,
            "f1": 2 * tp / denom if denom else 0.0,
        })

    recalculated = sum(item["f1"] for item in summary) / 4
    if not np.isclose(recalculated, metrics["parent_macro_f1"], atol=1e-10):
        raise RuntimeError("Inconsistent macro-F1 calculations")
    if not np.isclose(recalculated, checkpoint["parent_macro_f1"], atol=1e-10):
        raise RuntimeError("Saved checkpoint score does not match validation predictions")

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "validation_parent_predictions.csv"
    json_path = output / "validation_audit.json"
    pd.DataFrame(predictions).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "outer": args.outer, "inner": args.inner,
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_images": metrics["images"],
        "validation_parents": metrics["parents"],
        "parent_macro_f1": recalculated,
        "validation_loss": metrics["validation_loss"],
        "confusion_rows_actual_columns_predicted": confusion.tolist(),
        "per_class": summary,
        "misclassified_parents": [p for p in predictions if not p["correct"]],
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation images: {metrics['images']} | parents: {metrics['parents']}")
    print(f"Parent macro-F1: {recalculated:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted; class IDs 0..3):")
    for row in confusion:
        print(" ".join(f"{int(x):3d}" for x in row))
    print("Per-class parent support and F1:")
    for item in summary:
        print(f"class={item['class_id']} parents={item['parents']} f1={item['f1']:.4f}")
    print(f"Misclassified parent IDs: {[p['parent_measurement_id'] for p in predictions if not p['correct']]}")
    print(f"Prediction CSV: {csv_path}")
    print(f"Audit JSON: {json_path}")
    print("VALIDATION_224x224_AUDIT_PASS — outer test untouched")


if __name__ == "__main__":
    main()
