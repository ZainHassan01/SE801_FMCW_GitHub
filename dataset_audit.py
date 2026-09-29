import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATASET = r"C:\Users\Eng Zain\Desktop\Spring 26\ann\data_SAAB_SIRS_77GHz_FMCW.npy"
SEED = 2026
K = 5

data = np.load(DATASET, allow_pickle=True)

DRONES = {"D1", "D2", "D3", "D4", "D5", "D6"}
HUMANS = {"human_walk", "human_run"}
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


def map_nine(label):
    if label in DRONES:
        return label
    if label in HUMANS:
        return "Human"
    if label in BIRDS:
        return "Bird"
    if label.upper() == "CR":
        return "CR"
    return f"UNKNOWN::{label}"


def map_four(label):
    if label in DRONES:
        return "Drone"
    if label in HUMANS:
        return "Human"
    if label in BIRDS:
        return "Bird"
    if label.upper() == "CR":
        return "CR"
    return f"UNKNOWN::{label}"


records = []

for measurement_id in range(data.shape[0]):
    signal = np.asarray(data[measurement_id, 1])

    if signal.shape[0] == 1280:
        segment_count = signal.shape[1]
    elif signal.shape[1] == 1280:
        segment_count = signal.shape[0]
    else:
        raise ValueError(
            f"Invalid signal shape at measurement {measurement_id}: "
            f"{signal.shape}"
        )

    label = get_label(data[measurement_id, 0])
    provided_splits = np.asarray(
        data[measurement_id, 4]
    ).reshape(-1)

    unique_splits = np.unique(provided_splits)

    records.append({
        "measurement_id": measurement_id,
        "raw_label": label,
        "nine_class": map_nine(label),
        "four_class": map_four(label),
        "segment_count": segment_count,
        "provided_splits": "|".join(map(str, unique_splits)),
        "split_leakage": len(unique_splits) > 1,
    })

parents = pd.DataFrame(records)
parents.to_csv("parent_measurement_audit.csv", index=False)


def class_summary(label_column):
    return (
        parents.groupby(label_column)
        .agg(
            parent_measurements=("measurement_id", "nunique"),
            segments=("segment_count", "sum"),
        )
        .sort_index()
    )


def test_nested_cv(label_column):
    y = parents[label_column].to_numpy()
    groups = parents["measurement_id"].to_numpy()
    X = np.zeros((len(parents), 1))
    all_classes = set(y)

    outer_cv = StratifiedGroupKFold(
        n_splits=K,
        shuffle=True,
        random_state=SEED,
    )

    try:
        outer_splits = list(outer_cv.split(X, y, groups))
    except ValueError as error:
        return False, f"Outer split failed: {error}"

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_splits, start=1
    ):
        test_classes = set(y[test_idx])

        if test_classes != all_classes:
            missing = all_classes - test_classes
            return False, (
                f"Outer fold {outer_fold} missing classes: {missing}"
            )

        inner_y = y[train_idx]
        inner_groups = groups[train_idx]
        inner_X = np.zeros((len(train_idx), 1))

        inner_counts = pd.Series(inner_y).value_counts()

        if inner_counts.min() < K:
            return False, (
                f"Outer fold {outer_fold}: minimum training groups "
                f"per class = {inner_counts.min()}"
            )

        inner_cv = StratifiedGroupKFold(
            n_splits=K,
            shuffle=True,
            random_state=SEED,
        )

        try:
            inner_splits = list(
                inner_cv.split(inner_X, inner_y, inner_groups)
            )
        except ValueError as error:
            return False, (
                f"Outer fold {outer_fold}, inner split failed: {error}"
            )

        for inner_fold, (_, validation_idx) in enumerate(
            inner_splits, start=1
        ):
            validation_classes = set(inner_y[validation_idx])

            if validation_classes != all_classes:
                missing = all_classes - validation_classes
                return False, (
                    f"Outer {outer_fold}, inner {inner_fold} "
                    f"missing classes: {missing}"
                )

    return True, "All 5 outer × 5 inner folds passed"


print(f"Dataset shape: {data.shape}")
print(f"Parent measurements: {len(parents)}")
print(f"Total segments: {parents['segment_count'].sum()}")
print(
    "Parents crossing provided splits:",
    parents["split_leakage"].sum(),
)

for mapping in ["raw_label", "nine_class", "four_class"]:
    print(f"\n{'=' * 60}")
    print(mapping.upper())
    print(class_summary(mapping))

    passed, message = test_nested_cv(mapping)
    print(f"\nNested 5×5 CV: {'PASS' if passed else 'FAIL'}")
    print(message)