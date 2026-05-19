"""Measure correlations among the sentence/paragraph fact-check features.

The script reuses the 20 per-comment metrics from sentence_category_factcheck.py
and computes:
- a full 20x20 Pearson correlation matrix among the features
- Pearson correlation of each feature with the AI label
- a ranked table of the strongest feature pairs

This is useful for deciding which features are redundant before training a
tree-based model.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_category_factcheck import CATEGORIES, compute_comment_metrics


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


LABEL_TO_Y = {"HUMAN": 0.0, "AI": 1.0}
LABEL_COLORS = {0.0: "#1f77b4", 1.0: "#ff7f0e"}


def count_valid_rows(csv_path: Path) -> tuple[int, Counter[str]]:
    row_count = 0
    label_counts: Counter[str] = Counter()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label", "") or "").strip().upper()
            if label not in LABEL_TO_Y:
                continue
            row_count += 1
            label_counts[label] += 1

    return row_count, label_counts


def load_feature_matrix(csv_path: Path, row_count: int) -> tuple[np.ndarray, np.ndarray]:
    feature_keys = [category.metric_key for category in CATEGORIES]
    feature_matrix = np.empty((row_count, len(feature_keys)), dtype=np.float64)
    label_vector = np.empty(row_count, dtype=np.float64)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        index = 0
        for row in reader:
            label = (row.get("label", "") or "").strip().upper()
            if label not in LABEL_TO_Y:
                continue

            comment = row.get("comment", "") or ""
            metrics = compute_comment_metrics(comment)
            feature_matrix[index] = [metrics[key] for key in feature_keys]
            label_vector[index] = LABEL_TO_Y[label]
            index += 1

    return feature_matrix, label_vector


def pearson_correlation_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    sample_count = feature_matrix.shape[0]
    sums = feature_matrix.sum(axis=0)
    cross_products = feature_matrix.T @ feature_matrix
    covariance = (cross_products - np.outer(sums, sums) / sample_count) / (sample_count - 1)
    standard_deviations = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_deviations, standard_deviations)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator
    np.fill_diagonal(correlation, 1.0)
    return correlation


def pearson_correlation_with_label(feature_matrix: np.ndarray, label_vector: np.ndarray) -> np.ndarray:
    sample_count = feature_matrix.shape[0]
    sums_x = feature_matrix.sum(axis=0)
    sum_y = label_vector.sum()
    sums_xy = feature_matrix.T @ label_vector
    sum_y2 = float(np.square(label_vector).sum())

    covariance_xy = (sums_xy - sums_x * sum_y / sample_count) / (sample_count - 1)
    variance_x = (np.square(feature_matrix).sum(axis=0) - np.square(sums_x) / sample_count) / (sample_count - 1)
    variance_y = (sum_y2 - (sum_y * sum_y) / sample_count) / (sample_count - 1)
    std_x = np.sqrt(variance_x)
    std_y = float(np.sqrt(variance_y))

    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance_xy / (std_x * std_y)


def write_matrix_csv(output_path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{value:.12f}" for value in row]])


def write_pairwise_csv(output_path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            corr = float(matrix[i, j])
            rows.append((labels[i], labels[j], corr, abs(corr)))
    rows.sort(key=lambda item: item[3], reverse=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_a", "feature_b", "pearson_correlation", "absolute_correlation"])
        for feature_a, feature_b, corr, abs_corr in rows:
            writer.writerow([feature_a, feature_b, f"{corr:.12f}", f"{abs_corr:.12f}"])


def write_label_csv(output_path: Path, labels: list[str], correlations: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "pearson_correlation_with_AI_label"])
        for label, corr in sorted(zip(labels, correlations), key=lambda item: abs(float(item[1])), reverse=True):
            writer.writerow([label, f"{float(corr):.12f}"])


def plot_heatmap(output_path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Feature correlation matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_label_correlations(output_path: Path, labels: list[str], correlations: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(correlations)
    sorted_labels = [labels[i] for i in order]
    sorted_correlations = correlations[order]
    colors = [LABEL_COLORS[1.0 if corr >= 0 else 0.0] for corr in sorted_correlations]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_labels, sorted_correlations, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Pearson correlation with AI label")
    ax.set_title("Feature correlation with AI label")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute correlations among the 20 sentence features.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "train.csv", help="Training CSV path.")
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_feature_correlation_matrix.csv",
        help="CSV output for the full feature correlation matrix.",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_pairwise_correlations.csv",
        help="CSV output for ranked feature pairs.",
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_label_correlations.csv",
        help="CSV output for feature correlations with the AI label.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_feature_correlation_heatmap.png",
        help="Heatmap output for feature-feature correlations.",
    )
    parser.add_argument(
        "--label-plot-output",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_label_correlations.png",
        help="Bar chart output for label correlations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_count, label_counts = count_valid_rows(args.input)
    feature_matrix, label_vector = load_feature_matrix(args.input, row_count)
    labels = [category.title for category in CATEGORIES]

    matrix = pearson_correlation_matrix(feature_matrix)
    label_correlations = pearson_correlation_with_label(feature_matrix, label_vector)

    write_matrix_csv(args.matrix_output, matrix, labels)
    write_pairwise_csv(args.pairs_output, matrix, labels)
    write_label_csv(args.label_output, labels, label_correlations)
    plot_heatmap(args.heatmap_output, matrix, labels)
    plot_label_correlations(args.label_plot_output, labels, label_correlations)

    top_pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            corr = float(matrix[i, j])
            top_pairs.append((abs(corr), labels[i], labels[j], corr))
    top_pairs.sort(reverse=True)

    print(f"Analyzed {row_count:,} comments from {args.input}")
    print(f"Label counts: HUMAN={label_counts['HUMAN']:,}, AI={label_counts['AI']:,}")
    print("Top 10 absolute feature-feature correlations:")
    for _, feature_a, feature_b, corr in top_pairs[:10]:
        print(f"  {feature_a} vs {feature_b}: {corr:.3f}")

    print("Strongest label correlations (AI label):")
    label_order = np.argsort(np.abs(label_correlations))[::-1]
    for index in label_order[:10]:
        print(f"  {labels[index]}: {label_correlations[index]:.3f}")

    print(f"Saved matrix CSV to {args.matrix_output}")
    print(f"Saved pairwise CSV to {args.pairs_output}")
    print(f"Saved label CSV to {args.label_output}")
    print(f"Saved heatmap to {args.heatmap_output}")
    print(f"Saved label plot to {args.label_plot_output}")


if __name__ == "__main__":
    main()