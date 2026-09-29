"""Generate final Track-S2 figures, tables, and a Markdown results section."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "artifacts_224x224" / "cnn_final_nested_cv_summary"
OUTPUT = ROOT / "artifacts_224x224" / "cnn_final_report"
CLASS_NAMES = ["Drone", "Bird", "Human", "CR"]
CONDITION_LABELS = {
    "clean": "Clean",
    "gaussian_noise_0.05": "Noise σ=0.05",
    "gaussian_noise_0.10": "Noise σ=0.10",
    "time_mask_20pct": "Time mask 20%",
    "zero_doppler_mask_15pct": "Zero-Doppler mask 15%",
    "doppler_shift_8px": "Doppler shift 8 px",
    "contrast_half": "Contrast ×0.5",
}


def configure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
    })


def save_figure(fig, stem):
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def outer_fold_plot(fold_results, summary):
    clean = fold_results.loc[fold_results["condition"] == "clean"].sort_values(
        "outer_fold"
    )
    mean = summary["primary_nested_cv_result"]["mean"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ax.plot(
        clean["outer_fold"], clean["parent_macro_f1"],
        marker="o", linewidth=2, markersize=7, color="#1f4e79",
        label="Outer-fold score",
    )
    ax.axhline(mean, color="#c00000", linewidth=1.7, linestyle="--",
               label=f"Mean = {mean:.4f}")
    for x, y in zip(clean["outer_fold"], clean["parent_macro_f1"]):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlabel="Outer fold", ylabel="Parent-level macro-F1",
           title="Clean outer-test performance across nested-CV folds",
           xticks=range(1, 6), ylim=(0.70, 1.04))
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, "figure_1_outer_fold_clean_macro_f1")


def robustness_plot(condition_summary):
    block = condition_summary.copy()
    labels = [CONDITION_LABELS[value] for value in block["condition"]]
    values = block["fold_mean_parent_macro_f1"].to_numpy()
    errors = block["fold_std_parent_macro_f1"].to_numpy()
    colors = ["#2e7d32"] + ["#b24a3a"] * (len(block) - 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    positions = np.arange(len(block))
    bars = ax.bar(
        positions, values, yerr=errors, capsize=4, color=colors,
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(positions, labels, rotation=28, ha="right")
    ax.set(ylabel="Parent-level macro-F1",
           title="Representation-domain robustness", ylim=(0, 1.05))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "figure_2_robustness_macro_f1")


def degradation_plot(condition_summary):
    block = condition_summary.loc[
        condition_summary["condition"] != "clean"
    ].copy()
    block["label"] = block["condition"].map(CONDITION_LABELS)
    block = block.sort_values("mean_delta_from_clean")
    values = block["mean_delta_from_clean"].to_numpy()
    colors = ["#c00000" if value < -0.1 else "#4c78a8" for value in values]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    bars = ax.barh(block["label"], values, color=colors,
                   edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Mean change from clean macro-F1",
           title="Performance degradation under corruption", xlim=(-0.65, 0.05))
    for bar, value in zip(bars, values):
        ax.text(value - 0.01 if value < 0 else value + 0.01,
                bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                ha="right" if value < 0 else "left", va="center", fontsize=9)
    save_figure(fig, "figure_3_corruption_delta_from_clean")


def confusion_plot(matrix):
    matrix = np.asarray(matrix, dtype=int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_totals, 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(4):
        for column in range(4):
            color = "white" if normalized[row, column] > 0.55 else "black"
            ax.text(column, row,
                    f"{matrix[row, column]}\n{normalized[row, column] * 100:.1f}%",
                    ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(4), CLASS_NAMES)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set(xlabel="Predicted class", ylabel="Actual class",
           title="Pooled outer-test confusion matrix (130 parents)")
    fig.colorbar(image, ax=ax, label="Row-normalized proportion", shrink=0.86)
    save_figure(fig, "figure_4_pooled_confusion_matrix")


def class_f1_plot(summary):
    values = summary["pooled_out_of_fold_clean_result"]["per_class_parent_f1"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    bars = ax.bar(CLASS_NAMES, values, color="#4472c4",
                  edgecolor="black", linewidth=0.5)
    ax.set(ylabel="Parent-level F1", title="Pooled per-class performance",
           ylim=(0, 1.05))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value:.3f}", ha="center", fontsize=9)
    save_figure(fig, "figure_5_pooled_per_class_f1")


def calibration_plot(summary):
    pooled = summary["pooled_out_of_fold_clean_result"]
    raw = pooled["raw_calibration"]
    calibrated = pooled["temperature_scaled_calibration"]
    metric_keys = ["ece_10_bin", "brier", "nll"]
    titles = ["ECE (10 bins)", "Brier score", "Negative log-likelihood"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.2), constrained_layout=True)
    for ax, key, title in zip(axes, metric_keys, titles):
        values = [raw[key], calibrated[key]]
        bars = ax.bar(["Raw", "Calibrated"], values,
                      color=["#a5a5a5", "#70ad47"],
                      edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.04,
                    f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Pooled probability calibration", fontsize=11)
    save_figure(fig, "figure_6_calibration_comparison")


def write_results_section(summary, condition_summary):
    primary = summary["primary_nested_cv_result"]
    pooled = summary["pooled_out_of_fold_clean_result"]
    interval = pooled["parent_bootstrap_95_percent_interval"]
    raw = pooled["raw_calibration"]
    calibrated = pooled["temperature_scaled_calibration"]
    rows = []
    for _, row in condition_summary.iterrows():
        rows.append(
            f"| {CONDITION_LABELS[row['condition']]} | "
            f"{row['fold_mean_parent_macro_f1']:.4f} ± "
            f"{row['fold_std_parent_macro_f1']:.4f} | "
            f"{row['mean_delta_from_clean']:+.4f} |"
        )
    text = f"""# Track S2: CNN Results

## Evaluation protocol

Evaluation used leakage-safe 5 × 5 nested stratified cross-validation with parent measurement as the grouping unit and seed 2026. Latent-diversity sampling was selected independently within every outer fold using mean inner-validation parent macro-F1. Each final model was refitted on its complete outer-training partition using the rounded median selected epoch. Temperature scaling was fitted from inner out-of-fold parent logits before the corresponding outer-test set was accessed. Each outer-test parent was evaluated exactly once.

## Classification performance

The primary clean-data result was **{primary['mean']:.4f} ± {primary['standard_deviation']:.4f} parent macro-F1** across the five outer folds. Pooling the 130 unique out-of-fold parent predictions produced macro-F1 **{pooled['parent_macro_f1']:.4f}**. A 10,000-resample class-stratified parent bootstrap gave a 95% interval of **[{interval['lower_95_percentile']:.4f}, {interval['upper_95_percentile']:.4f}]**. The pooled estimate is reported as a complementary summary; the fold mean and standard deviation remain the primary nested-CV result.

## Calibration

Temperature scaling reduced pooled 10-bin ECE from **{raw['ece_10_bin']:.4f}** to **{calibrated['ece_10_bin']:.4f}**. Corresponding Brier scores were **{raw['brier']:.4f}** and **{calibrated['brier']:.4f}**, while negative log-likelihood changed from **{raw['nll']:.4f}** to **{calibrated['nll']:.4f}**.

## Robustness

| Condition | Macro-F1 (mean ± SD) | Mean change |
|---|---:|---:|
{chr(10).join(rows)}

The CNN was stable under a 20% temporal mask and an eight-pixel Doppler translation. In contrast, Gaussian noise, reduced contrast, and masking of the central 15% zero-Doppler region caused large performance losses. Together with the Grad-CAM audit, these results indicate reliance on absolute intensity and central zero-Doppler structure. This limitation should be reported directly rather than corrected after viewing the outer-test results.

## Reporting rule

No hyperparameter, sampling, preprocessing, calibration, or model decision was modified after outer-test evaluation. The outer results therefore remain a valid final nested-CV estimate.
"""
    (OUTPUT / "s2_results_section.md").write_text(text, encoding="utf-8")


def main():
    required = [
        INPUT / "s2_final_nested_cv_summary.json",
        INPUT / "nested_cv_condition_summary.csv",
        INPUT / "outer_fold_condition_results.csv",
        INPUT / "pooled_clean_confusion_matrix.csv",
        INPUT / "pooled_clean_parent_predictions.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final artifacts: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    with required[0].open(encoding="utf-8") as file:
        summary = json.load(file)
    condition_summary = pd.read_csv(required[1])
    fold_results = pd.read_csv(required[2])
    confusion = pd.read_csv(required[3], index_col=0).to_numpy(dtype=int)

    outer_fold_plot(fold_results, summary)
    robustness_plot(condition_summary)
    degradation_plot(condition_summary)
    confusion_plot(confusion)
    class_f1_plot(summary)
    calibration_plot(summary)
    write_results_section(summary, condition_summary)

    table = condition_summary[[
        "condition", "fold_mean_parent_macro_f1",
        "fold_std_parent_macro_f1", "mean_delta_from_clean",
        "mean_calibrated_ece_10_bin",
    ]].copy()
    table["condition"] = table["condition"].map(CONDITION_LABELS)
    table.to_csv(OUTPUT / "table_robustness_summary.csv", index=False)
    manifest = {
        "source_directory": str(INPUT),
        "output_directory": str(OUTPUT),
        "figures": [
            "figure_1_outer_fold_clean_macro_f1",
            "figure_2_robustness_macro_f1",
            "figure_3_corruption_delta_from_clean",
            "figure_4_pooled_confusion_matrix",
            "figure_5_pooled_per_class_f1",
            "figure_6_calibration_comparison",
        ],
        "formats": ["PNG 300 dpi", "vector PDF"],
        "primary_metric": "parent-level macro-F1",
    }
    (OUTPUT / "report_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created report artifacts: {OUTPUT}")
    print("Figures: 6 PNG + 6 PDF")
    print("Results section: s2_results_section.md")
    print("S2_REPORT_ARTIFACTS_PASS")


if __name__ == "__main__":
    main()
