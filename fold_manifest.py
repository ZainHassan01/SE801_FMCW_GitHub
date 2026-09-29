import csv
import json
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


DATASET = Path(
    r"C:\Users\Eng Zain\Desktop\Spring 26\ann"
    r"\data_SAAB_SIRS_77GHz_FMCW.npy"
)

OUTPUT_DIR = Path("manifests")
SEED = 2026
K = 5

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


def get_raw_label(value):
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


def sha256(filename):
    digest = hashlib.sha256()

    with open(filename, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


OUTPUT_DIR.mkdir(exist_ok=True)

data = np.load(DATASET, allow_pickle=True)

parents = []

for measurement_id in range(data.shape[0]):
    raw_label = get_raw_label(data[measurement_id, 0])
    mapped_class = map_class(raw_label)

    signal = np.asarray(data[measurement_id, 1])

    if signal.ndim != 2 or signal.shape[0] != 1280:
        raise ValueError(
            f"Measurement {measurement_id}: "
            f"invalid signal shape {signal.shape}"
        )

    parents.append({
        "measurement_id": measurement_id,
        "class_name": mapped_class,
        "class_id": CLASS_TO_ID[mapped_class],
        "segments": signal.shape[1],
    })

del data

measurement_ids = np.array([
    parent["measurement_id"]
    for parent in parents
])

labels = np.array([
    parent["class_name"]
    for parent in parents
])

X = np.zeros((len(parents), 1))

outer_cv = StratifiedGroupKFold(
    n_splits=K,
    shuffle=True,
    random_state=SEED,
)

outer_splits = list(
    outer_cv.split(
        X,
        labels,
        measurement_ids,
    )
)

manifest_path = OUTPUT_DIR / "fold_manifest.csv"

with manifest_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "class ID",
        "outer fold",
        "inner fold",
        "original-image identifier",
    ])

    expected_classes = set(CLASS_TO_ID)

    for outer_fold, (development_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        if set(labels[test_idx]) != expected_classes:
            raise RuntimeError(
                f"Outer fold {outer_fold} misses a class"
            )

        inner_labels = labels[development_idx]
        inner_groups = measurement_ids[development_idx]
        inner_X = np.zeros((len(development_idx), 1))

        inner_cv = StratifiedGroupKFold(
            n_splits=K,
            shuffle=True,
            random_state=SEED,
        )

        inner_splits = list(
            inner_cv.split(
                inner_X,
                inner_labels,
                inner_groups,
            )
        )

        inner_assignment = {}

        for inner_fold, (_, validation_idx) in enumerate(
            inner_splits,
            start=1,
        ):
            if set(inner_labels[validation_idx]) != expected_classes:
                raise RuntimeError(
                    f"Outer {outer_fold}, inner {inner_fold} "
                    f"misses a class"
                )

            for relative_idx in validation_idx:
                measurement_id = int(
                    inner_groups[relative_idx]
                )

                inner_assignment[measurement_id] = inner_fold

        test_measurements = set(
            measurement_ids[test_idx]
        )

        for parent in parents:
            measurement_id = parent["measurement_id"]

            if measurement_id in test_measurements:
                inner_fold = ""
            else:
                inner_fold = inner_assignment[measurement_id]

            for segment_id in range(parent["segments"]):
                filename = (
                    f"spectrograms/"
                    f"measurement_{measurement_id:03d}/"
                    f"segment_{segment_id:05d}.png"
                )

                writer.writerow([
                    filename,
                    parent["class_id"],
                    outer_fold,
                    inner_fold,
                    measurement_id,
                ])


class_mapping = {
    "class_to_id": CLASS_TO_ID,
    "id_to_class": {
        str(class_id): class_name
        for class_name, class_id in CLASS_TO_ID.items()
    },
    "raw_to_class": {
        **{label: "Drone" for label in sorted(DRONES)},
        **{label: "Bird" for label in sorted(BIRDS)},
        **{label: "Human" for label in sorted(HUMANS)},
        "CR": "CR",
    },
}

mapping_path = OUTPUT_DIR / "class_mapping.json"

with mapping_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        class_mapping,
        file,
        indent=4,
    )

manifest_hash = sha256(manifest_path)
mapping_hash = sha256(mapping_path)

with (OUTPUT_DIR / "checksums.sha256").open(
    "w",
    encoding="utf-8",
) as file:
    file.write(
        f"{manifest_hash}  fold_manifest.csv\n"
    )
    file.write(
        f"{mapping_hash}  class_mapping.json\n"
    )

print("Created:", manifest_path)
print("Created:", mapping_path)
print("Manifest rows:", K * sum(p["segments"] for p in parents))
print("fold_manifest.csv SHA-256:", manifest_hash)
print("class_mapping.json SHA-256:", mapping_hash)