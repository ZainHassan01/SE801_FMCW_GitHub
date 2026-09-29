import csv
import json
import shutil
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT / "data_SAAB_SIRS_77GHz_FMCW.npy"
)

SPECTROGRAM_DIR = (
    ROOT / "spectrograms_224x48"
)

OLD_MANIFEST_DIR = ROOT / "manifests"
OUTPUT_DIR = ROOT / "manifests_224x48"

OLD_FOLD_MANIFEST = (
    OLD_MANIFEST_DIR / "fold_manifest.csv"
)

OLD_CLASS_MAPPING = (
    OLD_MANIFEST_DIR / "class_mapping.json"
)

DATASET_MANIFEST = (
    OUTPUT_DIR / "dataset_manifest.csv"
)

FOLD_MANIFEST = (
    OUTPUT_DIR / "fold_manifest.csv"
)

CLASS_MAPPING = (
    OUTPUT_DIR / "class_mapping.json"
)

DATASET_CARD = (
    OUTPUT_DIR / "dataset_card.md"
)

CHECKSUM_FILE = (
    OUTPUT_DIR / "checksums.sha256"
)

EXPECTED_IMAGES = 75868
EXPECTED_FOLD_ROWS = EXPECTED_IMAGES * 5

CLASS_TO_ID = {
    "Drone": 0,
    "Bird": 1,
    "Human": 2,
    "CR": 3,
}

DRONES = {"D1", "D2", "D3", "D4", "D5", "D6"}

HUMANS = {
    "human_walk",
    "human_run",
}

BIRDS = {
    "black-headed gull",
    "heron",
    "pigeon",
    "raven",
    "seagull",
    "seagull and black-headed gull",
}


def get_label(value):
    value = np.asarray(value).squeeze()

    if isinstance(value, bytes):
        value = value.decode()

    return str(value).strip()


def map_class(raw_label):
    if raw_label in DRONES:
        return "Drone"

    if raw_label in BIRDS:
        return "Bird"

    if raw_label in HUMANS:
        return "Human"

    if raw_label.upper() == "CR":
        return "CR"

    raise ValueError(
        f"Unknown label: {raw_label}"
    )


def clean_value(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode()

    return value


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not OLD_FOLD_MANIFEST.is_file():
        raise FileNotFoundError(
            OLD_FOLD_MANIFEST
        )

    if not OLD_CLASS_MAPPING.is_file():
        raise FileNotFoundError(
            OLD_CLASS_MAPPING
        )

    # Preserve the already validated fold assignments.
    folds = pd.read_csv(
        OLD_FOLD_MANIFEST
    )

    folds["filename"] = (
        folds["filename"]
        .str.replace(
            r"^spectrograms/",
            "spectrograms_224x48/",
            regex=True,
        )
    )

    if len(folds) != EXPECTED_FOLD_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_FOLD_ROWS} fold rows, "
            f"found {len(folds)}"
        )

    if (
        folds["filename"].nunique()
        != EXPECTED_IMAGES
    ):
        raise ValueError(
            "Incorrect unique filename count"
        )

    folds.to_csv(
        FOLD_MANIFEST,
        index=False,
    )

    shutil.copy2(
        OLD_CLASS_MAPPING,
        CLASS_MAPPING,
    )

    print("Loading source dataset...")

    data = np.load(
        DATASET,
        allow_pickle=True,
    )

    class_parent_counts = defaultdict(int)
    class_segment_counts = defaultdict(int)

    total_segments = 0
    edge_fov_segments = 0

    with DATASET_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "class_id",
            "class_name",
            "original_label",
            "measurement_id",
            "segment_id",
            "range_metadata",
            "timestamp_seconds",
            "provided_split",
            "edge_fov",
            "file_size_bytes",
        ])

        for measurement_id in range(
            data.shape[0]
        ):
            raw_label = get_label(
                data[measurement_id, 0]
            )

            class_name = map_class(
                raw_label
            )

            class_id = CLASS_TO_ID[
                class_name
            ]

            signal = np.asarray(
                data[measurement_id, 1]
            )

            ranges = np.asarray(
                data[measurement_id, 2]
            ).reshape(-1)

            timestamps = np.asarray(
                data[measurement_id, 3]
            ).reshape(-1)

            provided_splits = np.asarray(
                data[measurement_id, 4]
            ).reshape(-1)

            edge_flags = np.asarray(
                data[measurement_id, 5]
            ).reshape(-1)

            segment_count = signal.shape[1]

            lengths = {
                segment_count,
                len(ranges),
                len(timestamps),
                len(provided_splits),
                len(edge_flags),
            }

            if len(lengths) != 1:
                raise ValueError(
                    f"Measurement {measurement_id}: "
                    f"metadata length mismatch"
                )

            class_parent_counts[
                class_name
            ] += 1

            class_segment_counts[
                class_name
            ] += segment_count

            for segment_id in range(
                segment_count
            ):
                filename = (
                    "spectrograms_224x48/"
                    f"measurement_{measurement_id:03d}/"
                    f"segment_{segment_id:05d}.png"
                )

                image_path = ROOT / filename

                if not image_path.is_file():
                    raise FileNotFoundError(
                        image_path
                    )

                edge_flag = int(
                    clean_value(
                        edge_flags[segment_id]
                    )
                )

                edge_fov_segments += int(
                    edge_flag != 0
                )

                writer.writerow([
                    filename,
                    class_id,
                    class_name,
                    raw_label,
                    measurement_id,
                    segment_id,
                    clean_value(
                        ranges[segment_id]
                    ),
                    clean_value(
                        timestamps[segment_id]
                    ),
                    clean_value(
                        provided_splits[segment_id]
                    ),
                    edge_flag,
                    image_path.stat().st_size,
                ])

                total_segments += 1

            print(
                f"\rManifest: "
                f"{measurement_id + 1}/"
                f"{data.shape[0]}",
                end="",
                flush=True,
            )

    print()

    if total_segments != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {total_segments}"
        )

    class_rows = []

    for class_name, class_id in sorted(
        CLASS_TO_ID.items(),
        key=lambda item: item[1],
    ):
        class_rows.append(
            f"| {class_id} | {class_name} | "
            f"{class_parent_counts[class_name]} | "
            f"{class_segment_counts[class_name]} |"
        )

    card = f"""# SE-801 77-GHz FMCW Dataset Card

## Dataset

This package contains micro-Doppler spectrograms derived from
complex-valued, range-compressed 77-GHz FMCW radar measurements.

- Independent parent measurements: 130
- Derived spectrograms: {total_segments}
- Image dimensions: 224 Doppler rows × 48 time columns
- Image format: 8-bit grayscale PNG
- Classification task: Drone, Bird, Human and CR

## Class distribution

| ID | Class | Parent measurements | Spectrograms |
|---:|---|---:|---:|
{chr(10).join(class_rows)}

## Independence and splitting

The parent `measurement_id` is the independent cross-validation
group. Every spectrogram derived from the same parent remains in
the same partition.

The author-provided segment splits are retained only as metadata
because all 130 parents cross those partitions.

- Outer folds: 5
- Inner folds: 5
- Splitter: StratifiedGroupKFold
- Seed: 2026

## Spectrogram processing

1. Reshape each segment into 5 range cells × 256 slow-time samples.
2. Retain slow-time indices 54 through 203.
3. Apply a 64-sample Hann window.
4. Use hop length 2 and FFT length 256.
5. Produce 44 temporal STFT frames.
6. Apply FFT shift to centre zero Doppler.
7. Sum power noncoherently across five range cells.
8. Place positive Doppler toward the top.
9. Convert to relative logarithmic power.
10. Clip the dynamic range to −60 through 0 dB.
11. Normalize intensity to [0,1].
12. Remove only the asymmetric Nyquist row.
13. Do not apply the inherited 5% image-border crop.
14. Resize to 224 Doppler rows × 48 time columns.

The rectangular representation preserves the short temporal
structure more faithfully than square 224×224 stretching.

## Fold normalization

Mean and standard deviation must be calculated from the active
training fold only. Validation and test images must use those
unchanged training-fold statistics.

## Known limitations

- Only 130 measurements are statistically independent.
- Human contains only 11 parent measurements.
- Segments within one parent are correlated.
- Segment counts are strongly class- and parent-imbalanced.
- Parent-balanced training sampling is required.
- Random segment-level splitting is prohibited.
- Raw complex measurements cannot be used directly by S1–S4.

## Edge-of-field-of-view

The package retains {edge_fov_segments} flagged edge-of-field-of-view
segments. Their flags are stored in `dataset_manifest.csv`.

## Required package files

- dataset_card.md
- dataset_manifest.csv
- fold_manifest.csv
- class_mapping.json
- checksums.sha256
- spectrogram_validation.json
- spectrogram_QA_montage.png
"""

    DATASET_CARD.write_text(
        card,
        encoding="utf-8",
    )

    del data

    checksum_targets = [
        DATASET_CARD,
        DATASET_MANIFEST,
        FOLD_MANIFEST,
        CLASS_MAPPING,
        OUTPUT_DIR
        / "spectrogram_validation.json",
        OUTPUT_DIR
        / "spectrogram_QA_montage.png",
    ]

    checksum_targets.extend(
        sorted(
            SPECTROGRAM_DIR.rglob("*.png")
        )
    )

    print(
        f"Generating checksums for "
        f"{len(checksum_targets)} files..."
    )

    with CHECKSUM_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for index, path in enumerate(
            checksum_targets,
            start=1,
        ):
            digest = sha256(path)
            relative_path = (
                path.relative_to(ROOT)
                .as_posix()
            )

            file.write(
                f"{digest}  {relative_path}\n"
            )

            if index % 5000 == 0:
                print(
                    f"Checksummed {index}/"
                    f"{len(checksum_targets)}"
                )

    print("\nPackage completed")
    print("Images:", total_segments)
    print(
        "Fold rows:",
        len(folds),
    )
    print(
        "Edge-FOV segments:",
        edge_fov_segments,
    )
    print(
        "Output:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()