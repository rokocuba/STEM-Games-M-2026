"""Analyze the correlation between two sentence features.

The default pair is:
- sentences per paragraph
- question mark present

It writes a CSV summary and a plot that shows the selected pair colored by
HUMAN vs AI label. This is the right tool when you want to inspect two
categories at once instead of the full correlation matrix.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_category_factcheck import CATEGORIES, compute_comment_metrics


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


LABELS = ("HUMAN", "AI")
LABEL_COLORS = {
    "HUMAN": "#1f77b4",
    "AI": "#ff7f0e",
}
DEFAULT_X_FEATURE = "question_mark_present"
DEFAULT_Y_FEATURE = "sentences_per_paragraph"


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def resolve_feature(name: str):
    normalized = normalize_name(name)
    for category in CATEGORIES:
        aliases = {
            normalize_name(category.metric_key),
            normalize_name(category.csv_label),
            normalize_name(category.title),
        }
        if normalized in aliases:
            return category
    available = ", ".join(category.metric_key for category in CATEGORIES)
    raise ValueError(f"Unknown feature '{name}'. Available keys: {available}")


def load_pair(csv_path: Path, x_key: str, y_key: str):
    x_values = []
    y_values = []
    labels = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label", "") or "").strip().upper()
            if label not in LABELS:
                continue

            metrics = compute_comment_metrics(row.get("comment", "") or "")
            x_values.append(metrics[x_key])
            y_values.append(metrics[y_key])
            labels.append(label)

    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), np.asarray(labels)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize(x: np.ndarray, y: np.ndarray, labels: np.ndarray):
    summary = {}
    summary["overall_r"] = pearson_r(x, y)
    for label in LABELS:
        mask = labels == label
        summary[f"{label.lower()}_r"] = pearson_r(x[mask], y[mask])
        summary[f"{label.lower()}_x_rate"] = float(np.mean(x[mask] > 0))
        summary[f"{label.lower()}_y_mean_x1"] = float(np.mean(y[mask][x[mask] > 0])) if np.any(x[mask] > 0) else float("nan")
        summary[f"{label.lower()}_y_mean_x0"] = float(np.mean(y[mask][x[mask] <= 0])) if np.any(x[mask] <= 0) else float("nan")

    summary["overall_x_rate"] = float(np.mean(x > 0))
    summary["overall_y_mean_x1"] = float(np.mean(y[x > 0])) if np.any(x > 0) else float("nan")
    summary["overall_y_mean_x0"] = float(np.mean(y[x <= 0])) if np.any(x <= 0) else float("nan")
    return summary


def write_summary_csv(output_path: Path, summary: dict[str, float], x_label: str, y_label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_x", "feature_y", "metric", "value"])
        writer.writerow([x_label, y_label, "overall_pearson_r", f"{summary['overall_r']:.12f}"])
        writer.writerow([x_label, y_label, "human_pearson_r", f"{summary['human_r']:.12f}"])
        writer.writerow([x_label, y_label, "ai_pearson_r", f"{summary['ai_r']:.12f}"])
        writer.writerow([x_label, y_label, "overall_x_rate", f"{summary['overall_x_rate']:.12f}"])
        writer.writerow([x_label, y_label, "human_x_rate", f"{summary['human_x_rate']:.12f}"])
        writer.writerow([x_label, y_label, "ai_x_rate", f"{summary['ai_x_rate']:.12f}"])
        writer.writerow([x_label, y_label, "overall_y_mean_when_x1", f"{summary['overall_y_mean_x1']:.12f}"])
        writer.writerow([x_label, y_label, "overall_y_mean_when_x0", f"{summary['overall_y_mean_x0']:.12f}"])
        writer.writerow([x_label, y_label, "human_y_mean_when_x1", f"{summary['human_y_mean_x1']:.12f}"])
        writer.writerow([x_label, y_label, "human_y_mean_when_x0", f"{summary['human_y_mean_x0']:.12f}"])
        writer.writerow([x_label, y_label, "ai_y_mean_when_x1", f"{summary['ai_y_mean_x1']:.12f}"])
        writer.writerow([x_label, y_label, "ai_y_mean_when_x0", f"{summary['ai_y_mean_x0']:.12f}"])


def plot_pair(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    x_label: str,
    y_label: str,
    summary: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    unique_x = set(np.unique(x))
    is_binary_x = unique_x.issubset({0.0, 1.0}) and len(unique_x) <= 2

    rng = np.random.default_rng(42)
    sample_size = min(20_000, len(x))
    sample_indices = rng.choice(len(x), size=sample_size, replace=False)

    if is_binary_x:
        jitter = rng.normal(0.0, 0.04, size=sample_size)
        sampled_x = x[sample_indices] + jitter
        sampled_y = y[sample_indices]
        sampled_labels = labels[sample_indices]

        for label in LABELS:
            mask = sampled_labels == label
            ax.scatter(
                sampled_x[mask],
                sampled_y[mask],
                s=6,
                alpha=0.18,
                color=LABEL_COLORS[label],
                label=label,
                edgecolors="none",
            )

        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([f"No {x_label}", f"Has {x_label}"], rotation=0)
        ax.axvline(0.0, color="#444444", linewidth=0.8, alpha=0.4)
        ax.axvline(1.0, color="#444444", linewidth=0.8, alpha=0.4)
        ax.set_xlim(-0.35, 1.35)
    else:
        for label in LABELS:
            mask = labels == label
            ax.scatter(
                x[mask],
                y[mask],
                s=6,
                alpha=0.18,
                color=LABEL_COLORS[label],
                label=label,
                edgecolors="none",
            )

    ax.set_xlabel(x_label.replace("_", " "))
    ax.set_ylabel(y_label.replace("_", " "))
    ax.set_title(f"{y_label.replace('_', ' ')} vs {x_label.replace('_', ' ')}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    ax.text(
        0.01,
        0.99,
        f"overall r = {summary['overall_r']:.3f}\nHUMAN r = {summary['human_r']:.3f}\nAI r = {summary['ai_r']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a pair of sentence features.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "train.csv", help="Training CSV path.")
    parser.add_argument("--x-feature", type=str, default=DEFAULT_X_FEATURE, help="Feature to use on the x-axis.")
    parser.add_argument("--y-feature", type=str, default=DEFAULT_Y_FEATURE, help="Feature to use on the y-axis.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "pair_sentences_per_paragraph_vs_question_mark_present.png",
        help="Plot output path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ARTIFACTS_DIR / "pair_sentences_per_paragraph_vs_question_mark_present_summary.csv",
        help="CSV summary output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_category = resolve_feature(args.x_feature)
    y_category = resolve_feature(args.y_feature)

    x, y, labels = load_pair(args.input, x_category.metric_key, y_category.metric_key)
    summary = summarize(x, y, labels)
    plot_pair(args.output, x, y, labels, x_category.csv_label, y_category.csv_label, summary)
    write_summary_csv(args.summary_csv, summary, x_category.csv_label, y_category.csv_label)

    print(f"Analyzed {len(x):,} comments from {args.input}")
    print(f"x feature: {x_category.title}")
    print(f"y feature: {y_category.title}")
    print(f"Overall Pearson r: {summary['overall_r']:.3f}")
    print(f"HUMAN Pearson r: {summary['human_r']:.3f}")
    print(f"AI Pearson r: {summary['ai_r']:.3f}")
    print(f"Saved plot to {args.output}")
    print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()