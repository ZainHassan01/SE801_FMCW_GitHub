import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
SPECTROGRAM_DIR = ROOT / "spectrograms"
FOLD_MANIFEST = ROOT / "manifests" / "fold_manifest.csv"
OUTPUT_DIR = ROOT / "manifests"

EXPECTED_IMAGES = 75868
EXPECTED_MANIFEST_ROWS = EXPECTED_IMAGES * 5
SAMPLE_SIZE = 1000


def main():
    folds = pd.read_csv(FOLD_MANIFEST)

    required_columns = {
        "filename",
        "class ID",
        "outer fold",
        "inner fold",
        "original-image identifier",
    }

    missing_columns = required_columns - set(folds.columns)

    if missing_columns:
        raise ValueError(
            f"Missing manifest columns: {missing_columns}"
        )

    if len(folds) != EXPECTED_MANIFEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MANIFEST_ROWS} manifest rows, "
            f"found {len(folds)}"
        )

    grouped = folds.groupby("filename")

    rows_per_file = grouped.size()
    outer_folds_per_file = grouped["outer fold"].nunique()
    outer_test_rows = grouped["inner fold"].apply(
        lambda values: values.isna().sum()
    )

    if not (rows_per_file == 5).all():
        raise ValueError(
            "Every filename must occur exactly five times"
        )

    if not (outer_folds_per_file == 5).all():
        raise ValueError(
            "Every filename must cover five outer folds"
        )

    if not (outer_test_rows == 1).all():
        raise ValueError(
            "Every filename must be outer-test exactly once"
        )

    expected_files = set(folds["filename"].unique())

    actual_paths = sorted(
        SPECTROGRAM_DIR.rglob("*.png")
    )

    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in actual_paths
    }

    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files

    invalid_files = []

    for index, path in enumerate(actual_paths, start=1):
        try:
            with Image.open(path) as image:
                if image.size != (224, 224):
                    invalid_files.append(
                        f"{path}: size={image.size}"
                    )

                if image.mode != "L":
                    invalid_files.append(
                        f"{path}: mode={image.mode}"
                    )

                image.verify()

        except Exception as error:
            invalid_files.append(
                f"{path}: {error}"
            )

        if index % 5000 == 0:
            print(
                f"Validated {index}/{len(actual_paths)}"
            )

    sample_count = min(
        SAMPLE_SIZE,
        len(actual_paths),
    )

    sample_indices = np.linspace(
        0,
        len(actual_paths) - 1,
        sample_count,
        dtype=int,
    )

    means = []
    standard_deviations = []
    minimums = []
    maximums = []
    constant_images = []

    for index in sample_indices:
        path = actual_paths[index]

        with Image.open(path) as image:
            pixels = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

        means.append(float(pixels.mean()))
        standard_deviations.append(
            float(pixels.std())
        )
        minimums.append(float(pixels.min()))
        maximums.append(float(pixels.max()))

        if pixels.std() == 0:
            constant_images.append(str(path))

    montage_paths = [
        actual_paths[index]
        for index in np.linspace(
            0,
            len(actual_paths) - 1,
            16,
            dtype=int,
        )
    ]

    montage = Image.new(
        "L",
        (224 * 4, 224 * 4),
    )

    for position, path in enumerate(montage_paths):
        with Image.open(path) as image:
            row = position // 4
            column = position % 4

            montage.paste(
                image,
                (column * 224, row * 224),
            )

    montage_path = (
        OUTPUT_DIR / "spectrogram_QA_montage.png"
    )

    montage.save(montage_path)

    passed = (
        len(actual_paths) == EXPECTED_IMAGES
        and not missing_files
        and not extra_files
        and not invalid_files
        and not constant_images
    )

    report = {
        "passed": passed,
        "expected_images": EXPECTED_IMAGES,
        "actual_images": len(actual_paths),
        "manifest_rows": len(folds),
        "unique_manifest_files": len(expected_files),
        "missing_files": len(missing_files),
        "extra_files": len(extra_files),
        "invalid_files": invalid_files[:20],
        "sample_size": sample_count,
        "sample_pixel_mean": float(np.mean(means)),
        "sample_pixel_std_mean": float(
            np.mean(standard_deviations)
        ),
        "sample_pixel_min": float(np.min(minimums)),
        "sample_pixel_max": float(np.max(maximums)),
        "constant_sample_images": constant_images,
        "qa_montage": str(montage_path),
    }

    report_path = (
        OUTPUT_DIR / "spectrogram_validation.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    print("\nValidation:", "PASS" if passed else "FAIL")
    print("Images:", len(actual_paths))
    print("Missing:", len(missing_files))
    print("Extra:", len(extra_files))
    print("Invalid:", len(invalid_files))
    print("Constant sample images:", len(constant_images))
    print("Report:", report_path)
    print("Montage:", montage_path)


if __name__ == "__main__":
    main()