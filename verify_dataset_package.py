import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "manifests"

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
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading manifests...")

    dataset = pd.read_csv(DATASET_MANIFEST)
    folds = pd.read_csv(FOLD_MANIFEST)

    if len(dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"dataset_manifest.csv contains {len(dataset)} rows"
        )

    if dataset["filename"].nunique() != EXPECTED_IMAGES:
        raise ValueError(
            "Duplicate filenames in dataset_manifest.csv"
        )

    if len(folds) != EXPECTED_FOLD_ROWS:
        raise ValueError(
            f"fold_manifest.csv contains {len(folds)} rows"
        )

    if folds["filename"].nunique() != EXPECTED_IMAGES:
        raise ValueError(
            "Incorrect unique filename count in fold_manifest.csv"
        )

    if set(dataset["class_id"].unique()) != EXPECTED_CLASSES:
        raise ValueError(
            "Incorrect class IDs in dataset_manifest.csv"
        )

    if set(folds["class ID"].unique()) != EXPECTED_CLASSES:
        raise ValueError(
            "Incorrect class IDs in fold_manifest.csv"
        )

    dataset_files = set(dataset["filename"])
    fold_files = set(folds["filename"])

    if dataset_files != fold_files:
        raise ValueError(
            "Dataset and fold manifests contain different filenames"
        )

    rows_per_file = folds.groupby("filename").size()

    if not (rows_per_file == 5).all():
        raise ValueError(
            "Each image must occur five times in fold_manifest.csv"
        )

    outer_counts = (
        folds.groupby("filename")["outer fold"]
        .nunique()
    )

    if not (outer_counts == 5).all():
        raise ValueError(
            "Each image must occur in all five outer contexts"
        )

    folds["inner fold check"] = (
        folds["inner fold"]
        .fillna(0)
        .astype(int)
    )

    group_consistency = (
        folds.groupby([
            "outer fold",
            "original-image identifier",
        ])["inner fold check"]
        .nunique()
    )

    if group_consistency.max() != 1:
        raise ValueError(
            "Parent-measurement fold inconsistency detected"
        )

    for outer_fold in range(1, 6):
        outer_data = folds[
            folds["outer fold"] == outer_fold
        ]

        outer_test = outer_data[
            outer_data["inner fold"].isna()
        ]

        if set(outer_test["class ID"]) != EXPECTED_CLASSES:
            raise ValueError(
                f"Outer fold {outer_fold} test set "
                f"does not contain all classes"
            )

        for inner_fold in range(1, 6):
            inner_validation = outer_data[
                outer_data["inner fold"] == inner_fold
            ]

            if (
                set(inner_validation["class ID"])
                != EXPECTED_CLASSES
            ):
                raise ValueError(
                    f"Outer {outer_fold}, inner {inner_fold} "
                    f"does not contain all classes"
                )

    print("Checking image files...")

    for index, row in enumerate(
        dataset.itertuples(index=False),
        start=1,
    ):
        image_path = ROOT / row.filename

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if image_path.stat().st_size != row.file_size_bytes:
            raise ValueError(
                f"File-size mismatch: {row.filename}"
            )

        if index % 5000 == 0:
            print(
                f"Files checked: {index}/{EXPECTED_IMAGES}"
            )

    print("Verifying SHA-256 checksums...")

    checksum_entries = []

    with CHECKSUM_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            expected_hash, relative_path = line.split(
                "  ",
                maxsplit=1,
            )

            checksum_entries.append(
                (expected_hash, relative_path)
            )

    expected_checksum_entries = EXPECTED_IMAGES + 6

    if len(checksum_entries) != expected_checksum_entries:
        raise ValueError(
            f"Expected {expected_checksum_entries} checksum entries, "
            f"found {len(checksum_entries)}"
        )

    for index, (
        expected_hash,
        relative_path,
    ) in enumerate(checksum_entries, start=1):

        path = ROOT / relative_path

        if not path.is_file():
            raise FileNotFoundError(path)

        actual_hash = sha256(path)

        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch: {relative_path}"
            )

        if index % 5000 == 0:
            print(
                f"Checksums verified: "
                f"{index}/{len(checksum_entries)}"
            )

    print("\nDATASET PACKAGE: PASS")
    print("Images:", EXPECTED_IMAGES)
    print("Fold rows:", EXPECTED_FOLD_ROWS)
    print("Classes:", sorted(EXPECTED_CLASSES))
    print("Parent grouping: PASS")
    print("Nested 5x5 stratification: PASS")
    print("Checksums:", len(checksum_entries))


if __name__ == "__main__":
    main()