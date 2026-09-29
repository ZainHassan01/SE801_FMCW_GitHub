import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT / "data_SAAB_SIRS_77GHz_FMCW.npy"
)

SPECTROGRAM_DIR = ROOT / "spectrograms"
MANIFEST_DIR = ROOT / "manifests"

DATASET_MANIFEST = (
    MANIFEST_DIR / "dataset_manifest.csv"
)

DATASET_CARD = (
    MANIFEST_DIR / "dataset_card.md"
)

CHECKSUM_FILE = (
    MANIFEST_DIR / "checksums.sha256"
)

EXPECTED_IMAGES = 75868

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

    raise ValueError(f"Unknown label: {raw_label}")


def clean_value(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode()

    return value


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading source dataset...")
    data = np.load(
        DATASET,
        allow_pickle=True,
    )

    class_parent_counts = defaultdict(int)
    class_segment_counts = defaultdict(int)

    raw_parent_counts = defaultdict(int)
    raw_segment_counts = defaultdict(int)

    total_segments = 0
    total_edge_fov = 0

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

        for measurement_id in range(data.shape[0]):
            raw_label = get_label(
                data[measurement_id, 0]
            )

            class_name = map_class(raw_label)
            class_id = CLASS_TO_ID[class_name]

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

            number_of_segments = signal.shape[1]

            metadata_lengths = {
                len(ranges),
                len(timestamps),
                len(provided_splits),
                len(edge_flags),
                number_of_segments,
            }

            if len(metadata_lengths) != 1:
                raise ValueError(
                    f"Measurement {measurement_id}: "
                    f"metadata length mismatch"
                )

            class_parent_counts[class_name] += 1
            class_segment_counts[class_name] += (
                number_of_segments
            )

            raw_parent_counts[raw_label] += 1
            raw_segment_counts[raw_label] += (
                number_of_segments
            )

            for segment_id in range(number_of_segments):
                relative_filename = (
                    f"spectrograms/"
                    f"measurement_{measurement_id:03d}/"
                    f"segment_{segment_id:05d}.png"
                )

                image_path = ROOT / relative_filename

                if not image_path.is_file():
                    raise FileNotFoundError(
                        image_path
                    )

                edge_fov = int(
                    clean_value(edge_flags[segment_id])
                )

                total_edge_fov += int(edge_fov != 0)

                writer.writerow([
                    relative_filename,
                    class_id,
                    class_name,
                    raw_label,
                    measurement_id,
                    segment_id,
                    clean_value(ranges[segment_id]),
                    clean_value(timestamps[segment_id]),
                    clean_value(
                        provided_splits[segment_id]
                    ),
                    edge_fov,
                    image_path.stat().st_size,
                ])

                total_segments += 1

            print(
                f"\rManifest: {measurement_id + 1}/"
                f"{data.shape[0]} measurements",
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

    raw_rows = []

    for raw_label in sorted(raw_parent_counts):
        raw_rows.append(
            f"| {raw_label} | "
            f"{raw_parent_counts[raw_label]} | "
            f"{raw_segment_counts[raw_label]} |"
        )

    dataset_card = f"""# SE-801 FMCW Micro-Doppler Dataset Card

## Dataset

The dataset is derived from complex-valued, range-compressed
77-GHz FMCW radar measurements distributed through Zenodo
record 5845259.

The source contains {len(data)} independent parent measurements
and {total_segments} correlated scan segments.

Raw complex arrays are used only to generate micro-Doppler
spectrogram images. They are not supplied directly to the S1-S4
learning architectures.

## Classification task

The revised task contains four classes.

| Class ID | Class | Parent measurements | Spectrograms |
|---:|---|---:|---:|
{chr(10).join(class_rows)}

The output layer of every mandatory S1-S4 baseline contains four
output neurons.

## Original-label distribution

| Original label | Parent measurements | Segments |
|---|---:|---:|
{chr(10).join(raw_rows)}

## Independent sampling unit

The parent measurement identified by `measurement_id` is the
independent sampling and cross-validation group.

All spectrograms derived from one parent measurement remain in
the same outer-test or inner-validation partition.

The author-provided segment-level split indicators are retained
only as metadata. They are not used because all 130 parent
measurements cross the provided partitions.

## Cross-validation

- Outer folds: 5
- Inner folds: 5
- Splitter: StratifiedGroupKFold
- Group identifier: measurement_id
- Common initial random seed: 2026
- Fold definition: manifests/fold_manifest.csv
- Class definition: manifests/class_mapping.json

In `fold_manifest.csv`, each image occurs once for each outer-fold
context. A blank inner-fold value denotes the active outer-test
partition. Values 1-5 identify the inner-validation assignment
within the corresponding outer development partition.

## Spectrogram generation

Each scan segment is processed as follows:

1. Reshape the 1280 complex samples into five range cells by
   256 slow-time samples.
2. Retain slow-time indices 54 through 203, producing 150 samples.
3. Apply a 64-sample Hann analysis window.
4. Use a hop length of 2 samples.
5. Use a 256-point complex FFT.
6. Generate 44 temporal STFT frames.
7. Apply FFT shift to centre zero Doppler.
8. Sum spectral power non-coherently across the five range cells.
9. Orient positive Doppler toward the top of the image.
10. Convert to image-relative logarithmic power.
11. Clip the dynamic range to -60 through 0 dB.
12. Normalize grayscale intensity to [0,1].
13. Remove the outer 5% border from every side.
14. Remove the asymmetric Nyquist row.
15. Resize using bilinear interpolation to 224 by 224 pixels.
16. Store the result as an 8-bit grayscale PNG.

Time is represented horizontally. Doppler frequency is represented
vertically. Zero Doppler lies between the two central image rows.

The central 15% of the normalized Doppler axis defines the nominal
zero-Doppler region for S1 physics-guided feature extraction.

## Fold-dependent normalization

No global dataset mean or standard deviation is stored.

For every active outer and inner fold:

1. Calculate mean and standard deviation using training images only.
2. Apply those statistics to the corresponding training,
   validation and test images.
3. Never calculate normalization statistics from validation or
   outer-test images.

## Edge-of-field-of-view metadata

The dataset contains {total_edge_fov} segments flagged as
edge-of-field-of-view observations.

These flags are preserved in `dataset_manifest.csv` for controlled
analysis. They must not be used as prediction inputs.

## Data-efficiency experiments

The S1 20%, 40%, 60%, 80% and 100% subsets must be constructed
from parent measurements within the active inner-training
partition.

Every retained subset must contain at least one parent measurement
from Drone, Bird, Human and CR. Individual mini-batches are not
required to contain every class.

## Known limitations

- There are only 130 statistically independent measurements.
- Human is the smallest mapped class with 11 parent measurements.
- Segment counts are highly imbalanced across parent measurements.
- Segment-level metrics overstate the effective sample size.
- Primary confidence intervals must resample parent measurements.
- Random segment splitting is prohibited.
- Raw complex data cannot be used directly by the learning models.

## Package contents

- `dataset_card.md`
- `dataset_manifest.csv`
- `fold_manifest.csv`
- `class_mapping.json`
- `checksums.sha256`
- `spectrogram_validation.json`
- `spectrogram_QA_montage.png`
- `spectrograms/`
"""

    DATASET_CARD.write_text(
        dataset_card,
        encoding="utf-8",
    )

    del data

    checksum_targets = [
        MANIFEST_DIR / "dataset_card.md",
        MANIFEST_DIR / "dataset_manifest.csv",
        MANIFEST_DIR / "fold_manifest.csv",
        MANIFEST_DIR / "class_mapping.json",
        MANIFEST_DIR / "spectrogram_validation.json",
        MANIFEST_DIR / "spectrogram_QA_montage.png",
    ]

    checksum_targets.extend(
        sorted(SPECTROGRAM_DIR.rglob("*.png"))
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
            digest = calculate_sha256(path)
            relative_path = path.relative_to(ROOT).as_posix()

            file.write(
                f"{digest}  {relative_path}\n"
            )

            if index % 5000 == 0:
                print(
                    f"Checksummed {index}/"
                    f"{len(checksum_targets)}"
                )

    print("\nPackage completed")
    print("Dataset manifest:", DATASET_MANIFEST)
    print("Dataset card:", DATASET_CARD)
    print("Checksums:", CHECKSUM_FILE)
    print("Images:", total_segments)
    print("Edge-FOV segments:", total_edge_fov)


if __name__ == "__main__":
    main()