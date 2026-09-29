import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "manifests_224x48"

DATASET_MANIFEST = (
    MANIFEST_DIR / "dataset_manifest.csv"
)

FOLD_MANIFEST = (
    MANIFEST_DIR / "fold_manifest.csv"
)

CLASS_MAPPING = (
    MANIFEST_DIR / "class_mapping.json"
)

DATASET_CARD = (
    MANIFEST_DIR / "dataset_card.md"
)

CHECKSUM_FILE = (
    MANIFEST_DIR / "checksums.sha256"
)

EXPECTED_IMAGES = 75868
EXPECTED_FOLD_ROWS = EXPECTED_IMAGES * 5
EXPECTED_CHECKSUMS = EXPECTED_IMAGES + 6
EXPECTED_CLASSES = {0, 1, 2, 3}


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
    required_files = [
        DATASET_MANIFEST,
        FOLD_MANIFEST,
        CLASS_MAPPING,
        DATASET_CARD,
        CHECKSUM_FILE,
        MANIFEST_DIR
        / "spectrogram_validation.json",
        MANIFEST_DIR
        / "spectrogram_QA_montage.png",
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading manifests...")

    dataset = pd.read_csv(
        DATASET_MANIFEST
    )

    folds = pd.read_csv(
        FOLD_MANIFEST
    )

    if len(dataset) != EXPECTED_IMAGES:
        raise ValueError(
            "Incorrect dataset manifest row count"
        )

    if (
        dataset["filename"].nunique()
        != EXPECTED_IMAGES
    ):
        raise ValueError(
            "Duplicate dataset filenames"
        )

    if len(folds) != EXPECTED_FOLD_ROWS:
        raise ValueError(
            "Incorrect fold manifest row count"
        )

    if (
        folds["filename"].nunique()
        != EXPECTED_IMAGES
    ):
        raise ValueError(
            "Incorrect fold filename count"
        )

    if (
        set(dataset["class_id"].unique())
        != EXPECTED_CLASSES
    ):
        raise ValueError(
            "Incorrect dataset class IDs"
        )

    if (
        set(folds["class ID"].unique())
        != EXPECTED_CLASSES
    ):
        raise ValueError(
            "Incorrect fold class IDs"
        )

    if (
        set(dataset["filename"])
        != set(folds["filename"])
    ):
        raise ValueError(
            "Dataset and fold filenames differ"
        )

    rows_per_file = (
        folds.groupby("filename").size()
    )

    if not (rows_per_file == 5).all():
        raise ValueError(
            "Each image must occur five times"
        )

    outer_counts = (
        folds.groupby("filename")[
            "outer fold"
        ].nunique()
    )

    if not (outer_counts == 5).all():
        raise ValueError(
            "Each image must cover five outer folds"
        )

    folds["inner_check"] = (
        folds["inner fold"]
        .fillna(0)
        .astype(int)
    )

    group_consistency = (
        folds.groupby([
            "outer fold",
            "original-image identifier",
        ])["inner_check"]
        .nunique()
    )

    if group_consistency.max() != 1:
        raise ValueError(
            "Parent fold inconsistency detected"
        )

    for outer_fold in range(1, 6):
        outer_rows = folds[
            folds["outer fold"]
            == outer_fold
        ]

        outer_test = outer_rows[
            outer_rows["inner fold"].isna()
        ]

        if (
            set(outer_test["class ID"])
            != EXPECTED_CLASSES
        ):
            raise ValueError(
                f"Outer fold {outer_fold} "
                f"misses a class"
            )

        for inner_fold in range(1, 6):
            validation = outer_rows[
                outer_rows["inner fold"]
                == inner_fold
            ]

            if (
                set(validation["class ID"])
                != EXPECTED_CLASSES
            ):
                raise ValueError(
                    f"Outer {outer_fold}, "
                    f"inner {inner_fold} "
                    f"misses a class"
                )

    print("Checking image files...")

    for index, row in enumerate(
        dataset.itertuples(index=False),
        start=1,
    ):
        image_path = ROOT / row.filename

        if not image_path.is_file():
            raise FileNotFoundError(
                image_path
            )

        if (
            image_path.stat().st_size
            != row.file_size_bytes
        ):
            raise ValueError(
                f"File-size mismatch: "
                f"{row.filename}"
            )

        if index % 5000 == 0:
            print(
                f"Files checked: "
                f"{index}/{EXPECTED_IMAGES}"
            )

    checksum_entries = []

    with CHECKSUM_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            expected_hash, relative_path = (
                line.split("  ", maxsplit=1)
            )

            checksum_entries.append(
                (
                    expected_hash,
                    relative_path,
                )
            )

    if (
        len(checksum_entries)
        != EXPECTED_CHECKSUMS
    ):
        raise ValueError(
            f"Expected {EXPECTED_CHECKSUMS} "
            f"checksums, found "
            f"{len(checksum_entries)}"
        )

    print("Verifying checksums...")

    for index, (
        expected_hash,
        relative_path,
    ) in enumerate(
        checksum_entries,
        start=1,
    ):
        path = ROOT / relative_path

        if not path.is_file():
            raise FileNotFoundError(path)

        if sha256(path) != expected_hash:
            raise ValueError(
                f"Checksum mismatch: "
                f"{relative_path}"
            )

        if index % 5000 == 0:
            print(
                f"Checksums verified: "
                f"{index}/"
                f"{len(checksum_entries)}"
            )

    print("\nDATASET PACKAGE: PASS")
    print("Images:", EXPECTED_IMAGES)
    print(
        "Fold rows:",
        EXPECTED_FOLD_ROWS,
    )
    print(
        "Parent grouping: PASS"
    )
    print(
        "Nested 5×5 CV: PASS"
    )
    print(
        "Checksums:",
        len(checksum_entries),
    )


if __name__ == "__main__":
    main()