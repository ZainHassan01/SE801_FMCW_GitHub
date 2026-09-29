"""Aggregate the five completed one-time Track-S2 outer-test evaluations."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/szain/SE801_FMCW")
ARTIFACTS = ROOT / "artifacts_224x224"
OUTPUT = ARTIFACTS / "cnn_final_nested_cv_summary"
SEED = 2026
BOOTSTRAPS = 10_000
CONDITIONS = [
    "clean",
    "gaussian_noise_0.05",
    "gaussian_noise_0.10",
    "time_mask_20pct",
    "zero_doppler_mask_15pct",
    "doppler_shift_8px",
    "contrast_half",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(targets, predictions):
    scores = []
    for class_id in range(4):
        tp = np.sum((targets == class_id) & (predictions == class_id))
        fp = np.sum((targets != class_id) & (predictions == class_id))
        fn = np.sum((targets == class_id) & (predictions != class_id))
        denominator = 2 * tp + fp + fn
        scores.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores)), scores


def confusion_matrix(targets, predictions):
    matrix = np.zeros((4, 4), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def probability_metrics(probabilities, targets, bins=10):
    one_hot = np.eye(4, dtype=np.float64)[targets]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    selected = np.clip(
        probabilities[np.arange(len(targets)), targets], 1e-12, 1.0
    )
    nll = float(-np.mean(np.log(selected)))
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (
                (confidence >= edges[index])
                & (confidence <= edges[index + 1])
            )
        else:
            mask = (
                (confidence >= edges[index])
                & (confidence < edges[index + 1])
            )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {"ece_10_bin": float(ece), "brier": brier, "nll": nll}


def stratified_parent_bootstrap(targets, predictions):
    rng = np.random.default_rng(SEED)
    class_indices = {
        class_id: np.flatnonzero(targets == class_id)
        for class_id in range(4)
    }
    if any(len(indices) == 0 for indices in class_indices.values()):
        raise RuntimeError("Cannot bootstrap with an absent class")
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for iteration in range(BOOTSTRAPS):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices.values()
        ])
        values[iteration] = macro_f1(
            targets[sampled], predictions[sampled]
        )[0]
    return {
        "method": "class-stratified bootstrap of complete parent measurements",
        "resamples": BOOTSTRAPS,
        "seed": SEED,
        "lower_95_percentile": float(np.percentile(values, 2.5)),
        "upper_95_percentile": float(np.percentile(values, 97.5)),
        "bootstrap_standard_error": float(values.std(ddof=1)),
    }


def main() -> None:
    condition_rows = []
    prediction_frames = []
    fold_hashes = {}
    for outer in range(1, 6):
        directory = ARTIFACTS / "cnn_outer_refit" / f"outer_{outer}"
        results_path = directory / "outer_test_results.json"
        ledger_path = directory / "outer_test_evaluation_ledger.json"
        predictions_path = directory / "outer_test_parent_predictions.csv"
        with ledger_path.open() as file:
            ledger = json.load(file)
        if ledger["status"] != "completed":
            raise RuntimeError(f"Outer fold {outer} evaluation is incomplete")
        if ledger["results_sha256"] != sha256(results_path):
            raise RuntimeError(f"Outer fold {outer} result hash mismatch")
        if list(ledger["conditions"]) != CONDITIONS:
            raise RuntimeError(f"Outer fold {outer} condition mismatch")
        with results_path.open() as file:
            report = json.load(file)
        if int(report["outer_fold"]) != outer:
            raise RuntimeError(f"Outer fold {outer} report mismatch")
        if report["protocol"]["outer_test_evaluations_per_fold"] != 1:
            raise RuntimeError("Outer-test evaluation-count violation")
        reports = report["conditions"]
        if [row["condition"] for row in reports] != CONDITIONS:
            raise RuntimeError(f"Outer fold {outer} report order mismatch")
        for row in reports:
            condition_rows.append({
                "outer_fold": outer,
                "condition": row["condition"],
                "parent_macro_f1": float(row["parent_macro_f1"]),
                "delta_from_clean": float(row["delta_from_clean"]),
                "raw_ece_10_bin": float(row["raw_ece_10_bin"]),
                "raw_brier": float(row["raw_brier"]),
                "raw_nll": float(row["raw_nll"]),
                "calibrated_ece_10_bin": float(row["calibrated_ece_10_bin"]),
                "calibrated_brier": float(row["calibrated_brier"]),
                "calibrated_nll": float(row["calibrated_nll"]),
            })
        predictions = pd.read_csv(predictions_path)
        predictions.insert(0, "outer_fold", outer)
        score, _ = macro_f1(
            predictions["class_id"].to_numpy(dtype=int),
            predictions["predicted_class_id"].to_numpy(dtype=int),
        )
        clean_score = float(reports[0]["parent_macro_f1"])
        if not math.isclose(score, clean_score, abs_tol=1e-10):
            raise RuntimeError(f"Outer fold {outer} clean score mismatch")
        prediction_frames.append(predictions)
        fold_hashes[str(outer)] = {
            "results_sha256": sha256(results_path),
            "predictions_sha256": sha256(predictions_path),
        }

    conditions = pd.DataFrame(condition_rows).sort_values(
        ["condition", "outer_fold"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        "parent_measurement_id"
    )
    if len(predictions) != 130:
        raise RuntimeError(f"Expected 130 independent parents; found {len(predictions)}")
    if predictions["parent_measurement_id"].duplicated().any():
        raise RuntimeError("A parent appears in more than one outer test fold")
    if set(predictions["class_id"].astype(int)) != {0, 1, 2, 3}:
        raise RuntimeError("Pooled predictions lack a mapped class")

    summary_rows = []
    for condition in CONDITIONS:
        block = conditions.loc[conditions["condition"] == condition]
        if len(block) != 5:
            raise RuntimeError(f"Expected five results for {condition}")
        summary_rows.append({
            "condition": condition,
            "fold_mean_parent_macro_f1": float(block["parent_macro_f1"].mean()),
            "fold_std_parent_macro_f1": float(block["parent_macro_f1"].std(ddof=1)),
            "minimum_fold_parent_macro_f1": float(block["parent_macro_f1"].min()),
            "maximum_fold_parent_macro_f1": float(block["parent_macro_f1"].max()),
            "mean_delta_from_clean": float(block["delta_from_clean"].mean()),
            "mean_raw_ece_10_bin": float(block["raw_ece_10_bin"].mean()),
            "mean_calibrated_ece_10_bin": float(
                block["calibrated_ece_10_bin"].mean()
            ),
            "mean_raw_brier": float(block["raw_brier"].mean()),
            "mean_calibrated_brier": float(block["calibrated_brier"].mean()),
            "mean_raw_nll": float(block["raw_nll"].mean()),
            "mean_calibrated_nll": float(block["calibrated_nll"].mean()),
        })
    condition_summary = pd.DataFrame(summary_rows)

    targets = predictions["class_id"].to_numpy(dtype=int)
    predicted = predictions["predicted_class_id"].to_numpy(dtype=int)
    pooled_f1, pooled_class_f1 = macro_f1(targets, predicted)
    matrix = confusion_matrix(targets, predicted)
    raw_probabilities = predictions[
        [f"raw_probability_{index}" for index in range(4)]
    ].to_numpy(dtype=np.float64)
    calibrated_probabilities = predictions[
        [f"calibrated_probability_{index}" for index in range(4)]
    ].to_numpy(dtype=np.float64)
    raw_calibration = probability_metrics(raw_probabilities, targets)
    calibrated_calibration = probability_metrics(calibrated_probabilities, targets)
    bootstrap = stratified_parent_bootstrap(targets, predicted)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(OUTPUT / "outer_fold_condition_results.csv", index=False)
    condition_summary.to_csv(
        OUTPUT / "nested_cv_condition_summary.csv", index=False
    )
    predictions.to_csv(
        OUTPUT / "pooled_clean_parent_predictions.csv", index=False
    )
    pd.DataFrame(
        matrix,
        index=[f"actual_{index}" for index in range(4)],
        columns=[f"predicted_{index}" for index in range(4)],
    ).to_csv(OUTPUT / "pooled_clean_confusion_matrix.csv")

    clean_summary = condition_summary.loc[
        condition_summary["condition"] == "clean"
    ].iloc[0]
    final = {
        "protocol": {
            "outer_folds": 5,
            "inner_folds": 5,
            "seed": SEED,
            "independent_unit": "parent measurement",
            "parents": 130,
            "selected_strategy": "latent diversity in all five outer folds",
            "outer_test_scored_once_per_fold": True,
            "post_test_model_updates": False,
        },
        "primary_nested_cv_result": {
            "metric": "outer-fold parent-level macro-F1",
            "fold_values": conditions.loc[
                conditions["condition"] == "clean", "parent_macro_f1"
            ].tolist(),
            "mean": float(clean_summary["fold_mean_parent_macro_f1"]),
            "standard_deviation": float(clean_summary["fold_std_parent_macro_f1"]),
        },
        "pooled_out_of_fold_clean_result": {
            "parent_macro_f1": pooled_f1,
            "per_class_parent_f1": pooled_class_f1,
            "confusion_matrix": matrix.tolist(),
            "raw_calibration": raw_calibration,
            "temperature_scaled_calibration": calibrated_calibration,
            "parent_bootstrap_95_percent_interval": bootstrap,
        },
        "robustness": summary_rows,
        "source_hashes": fold_hashes,
        "interpretation_note": (
            "The fold mean and standard deviation are the primary nested-CV "
            "estimate. The pooled 130-parent score and stratified parent "
            "bootstrap interval are complementary summaries."
        ),
    }
    final_path = OUTPUT / "s2_final_nested_cv_summary.json"
    temporary = OUTPUT / "s2_final_nested_cv_summary.json.tmp"
    temporary.write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(final_path)

    print(
        f"Primary clean parent macro-F1: "
        f"{final['primary_nested_cv_result']['mean']:.4f} +/- "
        f"{final['primary_nested_cv_result']['standard_deviation']:.4f}"
    )
    print(f"Pooled 130-parent macro-F1: {pooled_f1:.4f}")
    print(
        f"Stratified parent-bootstrap 95% interval: "
        f"[{bootstrap['lower_95_percentile']:.4f}, "
        f"{bootstrap['upper_95_percentile']:.4f}]"
    )
    print(
        f"Pooled ECE raw={raw_calibration['ece_10_bin']:.4f} "
        f"calibrated={calibrated_calibration['ece_10_bin']:.4f}"
    )
    for row in summary_rows:
        print(
            f"{row['condition']}: "
            f"{row['fold_mean_parent_macro_f1']:.4f} +/- "
            f"{row['fold_std_parent_macro_f1']:.4f}; "
            f"delta={row['mean_delta_from_clean']:+.4f}"
        )
    print(f"Summary: {final_path}")
    print("S2_FINAL_NESTED_CV_SUMMARY_PASS")


if __name__ == "__main__":
    main()
