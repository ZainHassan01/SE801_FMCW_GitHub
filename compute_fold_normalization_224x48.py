import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent

FOLD_MANIFEST = (
    ROOT
    / "manifests_224x48"
    / "fold_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts_224x48"
    / "common"
)

IMAGE_STATS_FILE = (
    OUTPUT_DIR
    / "image_statistics.csv"
)

NORMALIZATION_FILE = (
    OUTPUT_DIR
    / "fold_normalization.json"
)

EXPECTED_IMAGES = 75868
EXPECTED_SIZE = (48, 224)


def calculate_image_statistics():
    folds = pd.read_csv(
        FOLD_MANIFEST
    )

    filenames = sorted(
        folds["filename"].unique()
    )

    if len(filenames) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(filenames)}"
        )

    records = []

    for index, filename in enumerate(
        filenames,
        start=1,
    ):
        image_path = ROOT / filename

        with Image.open(image_path) as image:
            if image.size != EXPECTED_SIZE:
                raise ValueError(
                    f"{filename}: "
                    f"invalid size {image.size}"
                )

            pixels = np.asarray(
                image,
                dtype=np.float64,
            ) / 255.0

        records.append({
            "filename": filename,
            "pixel_count": pixels.size,
            "pixel_sum": float(
                pixels.sum()
            ),
            "pixel_sum_squared": float(
                np.square(pixels).sum()
            ),
        })

        if index % 5000 == 0:
            print(
                f"Image statistics: "
                f"{index}/{len(filenames)}"
            )

    statistics = pd.DataFrame(
        records
    )

    statistics.to_csv(
        IMAGE_STATS_FILE,
        index=False,
    )

    return statistics


def aggregate_statistics(rows):
    pixel_count = int(
        rows["pixel_count"].sum()
    )

    pixel_sum = float(
        rows["pixel_sum"].sum()
    )

    pixel_sum_squared = float(
        rows["pixel_sum_squared"].sum()
    )

    mean = pixel_sum / pixel_count

    variance = (
        pixel_sum_squared / pixel_count
        - mean ** 2
    )

    variance = max(
        variance,
        0.0,
    )

    return {
        "images": int(len(rows)),
        "parent_measurements": int(
            rows[
                "original-image identifier"
            ].nunique()
        ),
        "pixels": pixel_count,
        "mean": mean,
        "standard_deviation": (
            variance ** 0.5
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds = pd.read_csv(
        FOLD_MANIFEST
    )

    if IMAGE_STATS_FILE.exists():
        print(
            "Loading cached image statistics..."
        )

        image_statistics = pd.read_csv(
            IMAGE_STATS_FILE
        )

        if (
            len(image_statistics)
            != EXPECTED_IMAGES
        ):
            raise ValueError(
                "Cached statistics are incomplete"
            )

    else:
        print(
            "Calculating image statistics..."
        )

        image_statistics = (
            calculate_image_statistics()
        )

    merged = folds.merge(
        image_statistics,
        on="filename",
        how="left",
        validate="many_to_one",
    )

    if merged["pixel_count"].isna().any():
        raise ValueError(
            "Missing image statistics"
        )

    results = {
        "representation": "224x48",
        "pixel_input_scale": "[0,1]",
        "normalization_scope": (
            "active training fold only"
        ),
        "standard_deviation": "population",
        "seed": 2026,
        "inner_training": {},
        "outer_refit": {},
    }

    for outer_fold in range(1, 6):
        outer_rows = merged[
            merged["outer fold"]
            == outer_fold
        ]

        outer_test = outer_rows[
            outer_rows["inner fold"].isna()
        ]

        outer_development = outer_rows[
            outer_rows["inner fold"].notna()
        ]

        outer_key = (
            f"outer_{outer_fold}"
        )

        results["outer_refit"][
            outer_key
        ] = aggregate_statistics(
            outer_development
        )

        for inner_fold in range(1, 6):
            inner_validation = (
                outer_development[
                    outer_development[
                        "inner fold"
                    ] == inner_fold
                ]
            )

            inner_training = (
                outer_development[
                    outer_development[
                        "inner fold"
                    ] != inner_fold
                ]
            )

            train_files = set(
                inner_training["filename"]
            )

            validation_files = set(
                inner_validation["filename"]
            )

            test_files = set(
                outer_test["filename"]
            )

            if train_files & validation_files:
                raise RuntimeError(
                    "Train-validation overlap"
                )

            if train_files & test_files:
                raise RuntimeError(
                    "Train-test overlap"
                )

            if (
                validation_files
                & test_files
            ):
                raise RuntimeError(
                    "Validation-test overlap"
                )

            key = (
                f"outer_{outer_fold}_"
                f"inner_{inner_fold}"
            )

            statistics = (
                aggregate_statistics(
                    inner_training
                )
            )

            results["inner_training"][
                key
            ] = statistics

            print(
                f"{key}: "
                f"images={statistics['images']}, "
                f"parents="
                f"{statistics['parent_measurements']}, "
                f"mean={statistics['mean']:.6f}, "
                f"std="
                f"{statistics['standard_deviation']:.6f}"
            )

    with NORMALIZATION_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        "\nCreated:",
        NORMALIZATION_FILE,
    )


if __name__ == "__main__":
    main()