"""Compute leakage-safe normalization after common crop and 224x224 resize."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.preprocessing import preprocess_image


ROOT = Path(__file__).resolve().parent
FOLD_MANIFEST = ROOT / "manifests_224x48" / "fold_manifest.csv"
IMAGE_ROOT = ROOT / "spectrograms_native_255x44"
OUTPUT = ROOT / "artifacts_224x224" / "common" / "fold_normalization.json"
EXPECTED_IMAGES = 75_868
PIXELS_PER_IMAGE = 224 * 224
PATTERN = re.compile(r"(?:^|/)(measurement_\d+)/(segment_\d+\.png)$")


def image_key(filename: str) -> str:
    match = PATTERN.search(str(filename).replace("\\", "/"))
    if match is None:
        raise ValueError(f"Unsupported manifest filename: {filename}")
    return f"{match.group(1)}/{match.group(2)}"


def statistics(indices: np.ndarray, sums: np.ndarray,
               sums_of_squares: np.ndarray) -> tuple[float, float]:
    if len(indices) == 0:
        raise RuntimeError("Cannot compute statistics for an empty partition")
    pixel_count = int(len(indices)) * PIXELS_PER_IMAGE
    total = float(np.sum(sums[indices], dtype=np.float64))
    total_squared = float(np.sum(sums_of_squares[indices], dtype=np.float64))
    mean = total / pixel_count
    variance = max(total_squared / pixel_count - mean * mean, 0.0)
    standard_deviation = variance ** 0.5
    if not np.isfinite(mean) or not np.isfinite(standard_deviation):
        raise RuntimeError("Nonfinite normalization statistics")
    if standard_deviation <= 0:
        raise RuntimeError("Zero normalization standard deviation")
    return mean, standard_deviation


def selected_indices(rows: pd.DataFrame, key_to_index: dict[str, int]) -> np.ndarray:
    keys = rows["image_key"].tolist()
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate image within an active fold partition")
    try:
        return np.asarray([key_to_index[key] for key in keys], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(f"Fold image not found in native dataset: {error}") from error


def main() -> None:
    if not FOLD_MANIFEST.is_file():
        raise FileNotFoundError(FOLD_MANIFEST)
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(IMAGE_ROOT)
    common_file = ROOT / "src" / "common" / "preprocessing.py"
    if not common_file.is_file():
        raise FileNotFoundError(
            f"Place preprocessing.py at {common_file} before running this script"
        )

    folds = pd.read_csv(FOLD_MANIFEST)
    required = {
        "filename", "class ID", "outer fold", "inner fold",
        "original-image identifier",
    }
    if not required.issubset(folds.columns):
        raise RuntimeError(f"Missing manifest columns: {required - set(folds.columns)}")
    folds["image_key"] = folds["filename"].map(image_key)

    # Each image appears once in every outer-fold context. Use one context to
    # enumerate the physical images, then reuse its measurements for all folds.
    unique = (
        folds.loc[folds["outer fold"] == 1, ["image_key"]]
        .drop_duplicates()
        .sort_values("image_key")
        .reset_index(drop=True)
    )
    if len(unique) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} unique images; found {len(unique)}"
        )
    keys = unique["image_key"].tolist()
    key_to_index = {key: index for index, key in enumerate(keys)}
    if len(key_to_index) != EXPECTED_IMAGES:
        raise RuntimeError("Nonunique native image keys")

    sums = np.empty(EXPECTED_IMAGES, dtype=np.float64)
    sums_of_squares = np.empty(EXPECTED_IMAGES, dtype=np.float64)
    print("Calculating post-crop, post-resize image statistics...", flush=True)
    for index, key in enumerate(keys):
        pixels = preprocess_image(IMAGE_ROOT / key)[0]
        if pixels.shape != (224, 224):
            raise RuntimeError(f"Incorrect common-preprocessing shape for {key}")
        if not np.isfinite(pixels).all():
            raise RuntimeError(f"Nonfinite pixels in {key}")
        sums[index] = np.sum(pixels, dtype=np.float64)
        sums_of_squares[index] = np.sum(
            pixels.astype(np.float64) ** 2, dtype=np.float64
        )
        if (index + 1) % 2500 == 0:
            print(f"Image statistics: {index + 1}/{EXPECTED_IMAGES}", flush=True)

    result = {
        "protocol": {
            "source_geometry": [255, 44],
            "crop_fraction_each_side": 0.05,
            "cropped_geometry": [229, 40],
            "output_geometry": [224, 224],
            "resize": "PIL bilinear",
            "pixel_scale_before_standardization": "uint8 / 255.0",
            "standard_deviation": "population",
            "seed": 2026,
            "leakage_rule": "statistics use active training parent measurements only",
        },
        "inner_training": {},
        "outer_refit": {},
    }

    all_parents = set(folds["original-image identifier"].unique())
    for outer_fold in range(1, 6):
        active = folds.loc[folds["outer fold"] == outer_fold]
        outer_train = active.loc[active["inner fold"].notna()].copy()
        outer_test = active.loc[active["inner fold"].isna()].copy()
        train_parents = set(outer_train["original-image identifier"])
        test_parents = set(outer_test["original-image identifier"])
        if train_parents & test_parents or train_parents | test_parents != all_parents:
            raise RuntimeError(f"Invalid parent grouping in outer fold {outer_fold}")

        indices = selected_indices(outer_train, key_to_index)
        mean, std = statistics(indices, sums, sums_of_squares)
        outer_key = f"outer_{outer_fold}"
        result["outer_refit"][outer_key] = {
            "images": len(outer_train),
            "parents": len(train_parents),
            "mean": mean,
            "standard_deviation": std,
        }
        print(
            f"{outer_key} refit: images={len(outer_train)}, "
            f"parents={len(train_parents)}, mean={mean:.6f}, std={std:.6f}",
            flush=True,
        )

        for inner_fold in range(1, 6):
            inner_train = outer_train.loc[
                outer_train["inner fold"] != inner_fold
            ].copy()
            inner_validation = outer_train.loc[
                outer_train["inner fold"] == inner_fold
            ].copy()
            inner_train_parents = set(inner_train["original-image identifier"])
            validation_parents = set(inner_validation["original-image identifier"])
            if inner_train_parents & validation_parents:
                raise RuntimeError(
                    f"Parent leakage in outer {outer_fold}, inner {inner_fold}"
                )
            if inner_train_parents | validation_parents != train_parents:
                raise RuntimeError(
                    f"Incomplete parents in outer {outer_fold}, inner {inner_fold}"
                )
            indices = selected_indices(inner_train, key_to_index)
            mean, std = statistics(indices, sums, sums_of_squares)
            key = f"outer_{outer_fold}_inner_{inner_fold}"
            result["inner_training"][key] = {
                "images": len(inner_train),
                "parents": len(inner_train_parents),
                "mean": mean,
                "standard_deviation": std,
            }
            print(
                f"{key}: images={len(inner_train)}, "
                f"parents={len(inner_train_parents)}, "
                f"mean={mean:.6f}, std={std:.6f}",
                flush=True,
            )

    if len(result["inner_training"]) != 25 or len(result["outer_refit"]) != 5:
        raise RuntimeError("Incomplete normalization ledger")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(f"\nCreated: {OUTPUT}")
    print("FOLD_NORMALIZATION_224x224_PASS")


if __name__ == "__main__":
    main()
