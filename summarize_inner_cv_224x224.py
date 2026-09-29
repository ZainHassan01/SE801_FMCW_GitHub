"""Aggregate all 25 paired S2 inner-fold runs without evaluating outer tests."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/szain/SE801_FMCW")
ARTIFACTS = ROOT / "artifacts_224x224"
OUTPUT = ARTIFACTS / "cnn_inner_cv_summary"


def best_epoch(history_path):
    with history_path.open() as file:
        data = json.load(file)
    epochs = data["epochs"]
    if not epochs:
        raise RuntimeError(f"No epochs in {history_path}")
    best = sorted(
        epochs,
        key=lambda row: (
            -float(row["parent_macro_f1"]),
            float(row["validation_loss"]),
            int(row["epoch"]),
        ),
    )[0]
    return {
        "epoch": int(best["epoch"]),
        "parent_macro_f1": float(best["parent_macro_f1"]),
        "validation_loss": float(best["validation_loss"]),
    }


def main():
    rows = []
    for outer in range(1, 6):
        for inner in range(1, 6):
            baseline_dir = ARTIFACTS / "cnn_inner" / f"outer_{outer}_inner_{inner}"
            diversity_dir = (
                ARTIFACTS / "cnn_latent_diversity" /
                f"outer_{outer}_inner_{inner}"
            )
            baseline = best_epoch(baseline_dir / "history.json")
            diversity = best_epoch(diversity_dir / "history.json")
            with (diversity_dir / "latent_diversity_construction.json").open() as file:
                construction = json.load(file)
            if construction["construction_partition"] != "active inner-training partition only":
                raise RuntimeError("Invalid latent construction partition")
            if construction["validation_embeddings_used"] or construction["outer_test_embeddings_used"]:
                raise RuntimeError("Leakage detected in latent construction ledger")
            difference = diversity["parent_macro_f1"] - baseline["parent_macro_f1"]
            if difference > 1e-12:
                winner = "latent_diversity"
            elif difference < -1e-12:
                winner = "baseline"
            else:
                winner = "tie"
            rows.append({
                "outer_fold": outer,
                "inner_fold": inner,
                "baseline_best_epoch": baseline["epoch"],
                "baseline_parent_macro_f1": baseline["parent_macro_f1"],
                "baseline_validation_loss": baseline["validation_loss"],
                "diversity_best_epoch": diversity["epoch"],
                "diversity_parent_macro_f1": diversity["parent_macro_f1"],
                "diversity_validation_loss": diversity["validation_loss"],
                "macro_f1_difference_diversity_minus_baseline": difference,
                "winner": winner,
                "training_parents": int(construction["training_parents"]),
                "selected_segments": int(construction["selected_segments"]),
                "full_mean_parent_cosine": float(construction["mean_full_parent_cosine"]),
                "selected_mean_parent_cosine": float(construction["mean_selected_parent_cosine"]),
                "cosine_redundancy_change": (
                    float(construction["mean_selected_parent_cosine"]) -
                    float(construction["mean_full_parent_cosine"])
                ),
            })

    frame = pd.DataFrame(rows).sort_values(["outer_fold", "inner_fold"])
    if len(frame) != 25:
        raise RuntimeError(f"Expected 25 paired runs; found {len(frame)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "paired_inner_fold_results.csv", index=False)

    selections = []
    for outer, block in frame.groupby("outer_fold", sort=True):
        baseline_mean = float(block["baseline_parent_macro_f1"].mean())
        diversity_mean = float(block["diversity_parent_macro_f1"].mean())
        if diversity_mean > baseline_mean + 1e-12:
            selected = "latent_diversity"
        elif baseline_mean > diversity_mean + 1e-12:
            selected = "baseline"
        else:
            # Strict tie: prefer the lower-complexity baseline sampler.
            selected = "baseline"
        epoch_column = (
            "diversity_best_epoch" if selected == "latent_diversity"
            else "baseline_best_epoch"
        )
        median_epoch = float(np.median(block[epoch_column].to_numpy()))
        refit_epochs = int(math.floor(median_epoch + 0.5))
        selections.append({
            "outer_fold": int(outer),
            "selected_sampling_strategy": selected,
            "selection_rule": (
                "highest mean parent macro-F1 across five inner folds; "
                "the lower-complexity baseline wins an exact tie"
            ),
            "baseline_mean_parent_macro_f1": baseline_mean,
            "baseline_std_parent_macro_f1": float(
                block["baseline_parent_macro_f1"].std(ddof=1)
            ),
            "latent_diversity_mean_parent_macro_f1": diversity_mean,
            "latent_diversity_std_parent_macro_f1": float(
                block["diversity_parent_macro_f1"].std(ddof=1)
            ),
            "mean_difference_diversity_minus_baseline": diversity_mean - baseline_mean,
            "selected_refit_epochs": refit_epochs,
            "epoch_rule": (
                "E_outer = floor(median(e_inner_1,...,e_inner_5) + 0.5), "
                "where each e_inner is the selected method's best epoch"
            ),
        })

    wins = frame["winner"].value_counts().to_dict()
    overall = {
        "paired_inner_folds": 25,
        "outer_folds": 5,
        "baseline_mean_parent_macro_f1": float(
            frame["baseline_parent_macro_f1"].mean()
        ),
        "baseline_std_across_inner_folds": float(
            frame["baseline_parent_macro_f1"].std(ddof=1)
        ),
        "latent_diversity_mean_parent_macro_f1": float(
            frame["diversity_parent_macro_f1"].mean()
        ),
        "latent_diversity_std_across_inner_folds": float(
            frame["diversity_parent_macro_f1"].std(ddof=1)
        ),
        "mean_paired_difference": float(
            frame["macro_f1_difference_diversity_minus_baseline"].mean()
        ),
        "median_paired_difference": float(
            frame["macro_f1_difference_diversity_minus_baseline"].median()
        ),
        "latent_diversity_wins": int(wins.get("latent_diversity", 0)),
        "baseline_wins": int(wins.get("baseline", 0)),
        "ties": int(wins.get("tie", 0)),
        "mean_cosine_redundancy_change": float(
            frame["cosine_redundancy_change"].mean()
        ),
        "outer_test_evaluated": False,
        "statistical_note": (
            "The 25 inner folds are not treated as 25 independent experiments. "
            "Model selection is performed separately within each outer fold."
        ),
    }
    result = {
        "overall_descriptive_summary": overall,
        "outer_fold_model_selection": selections,
        "pre_outer_test_calibration_plan": {
            "method": "scalar temperature scaling",
            "fitting_logits": (
                "out-of-fold parent logits produced by the selected method's five "
                "inner-validation checkpoints within the current outer fold"
            ),
            "leakage_control": (
                "each calibration parent is predicted by a checkpoint whose training "
                "partition excluded that parent; outer-test logits are not used"
            ),
            "freeze_rule": (
                "fit and save one temperature per outer fold before loading its outer-test images"
            ),
        },
        "outer_test_corruption_plan": {
            "reference": "clean parent-level macro-F1",
            "fixed_corruptions": [
                "gaussian_noise_0.05",
                "gaussian_noise_0.10",
                "time_mask_20pct",
                "zero_doppler_mask_15pct",
                "doppler_shift_8px",
                "contrast_half",
            ],
            "freeze_rule": (
                "corruption definitions and severities are frozen before outer-test evaluation"
            ),
        },
    }
    (OUTPUT / "inner_cv_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(selections).to_csv(
        OUTPUT / "outer_fold_model_selection.csv", index=False
    )

    print(f"Paired inner folds: {overall['paired_inner_folds']}")
    print(
        f"Baseline macro-F1: {overall['baseline_mean_parent_macro_f1']:.4f} "
        f"+/- {overall['baseline_std_across_inner_folds']:.4f}"
    )
    print(
        f"Latent diversity macro-F1: "
        f"{overall['latent_diversity_mean_parent_macro_f1']:.4f} +/- "
        f"{overall['latent_diversity_std_across_inner_folds']:.4f}"
    )
    print(
        f"Wins/ties/losses for diversity: "
        f"{overall['latent_diversity_wins']}/{overall['ties']}/{overall['baseline_wins']}"
    )
    print(
        f"Mean cosine redundancy change: "
        f"{overall['mean_cosine_redundancy_change']:+.4f}"
    )
    for row in selections:
        print(
            f"outer={row['outer_fold']} selected={row['selected_sampling_strategy']} "
            f"refit_epochs={row['selected_refit_epochs']}"
        )
    print(f"Summary directory: {OUTPUT}")
    print("INNER_CV_224x224_SUMMARY_PASS — outer tests untouched")


if __name__ == "__main__":
    main()
