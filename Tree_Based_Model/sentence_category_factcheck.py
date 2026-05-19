"""Fact-check sentence and paragraph heuristics against data/train.csv.

This script compares HUMAN and AI comments across 20 sentence/paragraph
features, generates a grid overview plot, saves an individual plot for each
feature, and writes a CSV summary with support verdicts.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


LABELS = ("HUMAN", "AI")
LABEL_COLORS = {
    "HUMAN": "#1f77b4",
    "AI": "#ff7f0e",
}
VERDICT_COLORS = {
    "supported": "#2e7d32",
    "contradicted": "#c62828",
    "tie": "#6a6a6a",
}

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

WORD_PATTERNS = {
    "although": re.compile(r"\balthough\b", re.IGNORECASE),
    "however": re.compile(r"\bHowever\b"),
    "but": re.compile(r"\bbut\b", re.IGNORECASE),
    "because": re.compile(r"\bbecause\b", re.IGNORECASE),
    "this": re.compile(r"\bthis\b", re.IGNORECASE),
    "others_or_researchers": re.compile(r"\b(?:others|researchers)\b", re.IGNORECASE),
    "numbers": re.compile(r"\d"),
    "et": re.compile(r"\bet\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class CategorySpec:
    number: int
    title: str
    csv_label: str
    claim_label: str
    metric_name: str
    metric_key: str
    slug: str


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    return parts if parts else [cleaned]


def split_paragraphs(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(cleaned) if part.strip()]
    return parts if parts else [cleaned]


def compute_comment_metrics(text: str) -> dict[str, float]:
    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)
    sentence_word_lengths: list[int] = []
    for sentence in sentences:
        sentence_tokens = tokenize_words(sentence)
        if sentence_tokens:
            sentence_word_lengths.append(len(sentence_tokens))
    paragraph_sentence_counts = [max(1, len(split_sentences(paragraph))) for paragraph in paragraphs] if paragraphs else []
    paragraph_word_counts = [len(tokenize_words(paragraph)) for paragraph in paragraphs] if paragraphs else []
    uppercase_count = sum(character.isupper() for character in text)
    period_count = text.count(".")

    metrics: dict[str, float] = {}
    metrics["sentences_per_paragraph"] = (
        sum(paragraph_sentence_counts) / len(paragraph_sentence_counts) if paragraph_sentence_counts else 0.0
    )
    metrics["words_per_paragraph"] = (
        sum(paragraph_word_counts) / len(paragraph_word_counts) if paragraph_word_counts else 0.0
    )
    metrics["right_parenthesis_present"] = float(")" in text)
    metrics["hyphen_present"] = float("-" in text)
    metrics["semicolon_or_colon_present"] = float(";" in text or ":" in text)
    metrics["question_mark_present"] = float("?" in text)
    metrics["apostrophe_present"] = float("'" in text)
    metrics["sentence_length_stddev"] = (
        statistics.pstdev(sentence_word_lengths) if len(sentence_word_lengths) >= 2 else 0.0
    )
    metrics["consecutive_sentence_gap"] = (
        sum(abs(current - previous) for previous, current in zip(sentence_word_lengths, sentence_word_lengths[1:]))
        / (len(sentence_word_lengths) - 1)
        if len(sentence_word_lengths) >= 2
        else 0.0
    )
    metrics["short_sentence_present"] = float(any(length < 11 for length in sentence_word_lengths))
    metrics["long_sentence_present"] = float(any(length > 34 for length in sentence_word_lengths))
    metrics["contains_although"] = float(bool(WORD_PATTERNS["although"].search(text)))
    metrics["contains_however"] = float(bool(WORD_PATTERNS["however"].search(text)))
    metrics["contains_but"] = float(bool(WORD_PATTERNS["but"].search(text)))
    metrics["contains_because"] = float(bool(WORD_PATTERNS["because"].search(text)))
    metrics["contains_this"] = float(bool(WORD_PATTERNS["this"].search(text)))
    metrics["contains_others_or_researchers"] = float(bool(WORD_PATTERNS["others_or_researchers"].search(text)))
    metrics["contains_numbers"] = float(bool(WORD_PATTERNS["numbers"].search(text)))
    metrics["uppercase_ge_2x_periods"] = float(uppercase_count >= 2 * period_count)
    metrics["contains_et"] = float(bool(WORD_PATTERNS["et"].search(text)))
    return metrics


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(1, "Sentences per paragraph", "sentences per paragraph", "HUMAN", "mean sentences/paragraph", "sentences_per_paragraph", "sentences_per_paragraph"),
    CategorySpec(2, "Words per paragraph", "words per paragraph", "HUMAN", "mean words/paragraph", "words_per_paragraph", "words_per_paragraph"),
    CategorySpec(3, "Right parenthesis present", "right parenthesis used", "HUMAN", "share of comments", "right_parenthesis_present", "right_parenthesis_present"),
    CategorySpec(4, "Hyphen present", "hyphen used", "HUMAN", "share of comments", "hyphen_present", "hyphen_present"),
    CategorySpec(5, "; or : present", "semicolon or colon used", "HUMAN", "share of comments", "semicolon_or_colon_present", "semicolon_or_colon_present"),
    CategorySpec(6, "Question mark present", "question mark used", "HUMAN", "share of comments", "question_mark_present", "question_mark_present"),
    CategorySpec(7, "Apostrophe present", "apostrophe used", "AI", "share of comments", "apostrophe_present", "apostrophe_present"),
    CategorySpec(8, "Sentence length std dev", "sentence length standard deviation", "HUMAN", "mean std dev", "sentence_length_stddev", "sentence_length_stddev"),
    CategorySpec(9, "Consecutive sentence gap", "consecutive sentence length gap", "HUMAN", "mean gap", "consecutive_sentence_gap", "consecutive_sentence_gap"),
    CategorySpec(10, "Any sentence <11 words", "any sentence under 11 words", "HUMAN", "share of comments", "short_sentence_present", "short_sentence_present"),
    CategorySpec(11, "Any sentence >34 words", "any sentence over 34 words", "HUMAN", "share of comments", "long_sentence_present", "long_sentence_present"),
    CategorySpec(12, "Contains although", "contains although", "HUMAN", "share of comments", "contains_although", "contains_although"),
    CategorySpec(13, "Contains However", "contains however", "HUMAN", "share of comments", "contains_however", "contains_however"),
    CategorySpec(14, "Contains but", "contains but", "HUMAN", "share of comments", "contains_but", "contains_but"),
    CategorySpec(15, "Contains because", "contains because", "HUMAN", "share of comments", "contains_because", "contains_because"),
    CategorySpec(16, "Contains this", "contains this", "HUMAN", "share of comments", "contains_this", "contains_this"),
    CategorySpec(17, "Contains others/researchers", "contains others or researchers", "AI", "share of comments", "contains_others_or_researchers", "contains_others_or_researchers"),
    CategorySpec(18, "Contains numbers", "contains numbers", "HUMAN", "share of comments", "contains_numbers", "contains_numbers"),
    CategorySpec(19, "Uppercase >= 2x periods", "uppercase at least 2x periods", "HUMAN", "share of comments", "uppercase_ge_2x_periods", "uppercase_ge_2x_periods"),
    CategorySpec(20, "Contains et", "contains et", "HUMAN", "share of comments", "contains_et", "contains_et"),
)


def analyze_dataset(csv_path: Path) -> tuple[Counter[str], dict[str, dict[int, float]]]:
    label_counts: Counter[str] = Counter()
    label_sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"comment", "label"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain comment and label columns")

        for row in reader:
            comment = row.get("comment", "") or ""
            label = (row.get("label", "") or "").strip().upper()
            if label not in LABELS:
                continue

            label_counts[label] += 1
            metrics = compute_comment_metrics(comment)
            for category in CATEGORIES:
                label_sums[label][category.number] += metrics[category.metric_key]

    return label_counts, label_sums


def summarize_results(
    label_counts: Counter[str],
    label_sums: dict[str, dict[int, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for category in CATEGORIES:
        human_mean = label_sums["HUMAN"][category.number] / label_counts["HUMAN"]
        ai_mean = label_sums["AI"][category.number] / label_counts["AI"]
        claim_mean = human_mean if category.claim_label == "HUMAN" else ai_mean
        other_mean = ai_mean if category.claim_label == "HUMAN" else human_mean

        if abs(claim_mean - other_mean) < 1e-12:
            verdict = "tie"
        elif claim_mean > other_mean:
            verdict = "supported"
        else:
            verdict = "contradicted"

        rows.append(
            {
                "category": category.number,
                "title": category.title,
                "label": category.csv_label,
                "claim_label": category.claim_label,
                "metric_name": category.metric_name,
                "human_frequency": human_mean,
                "ai_frequency": ai_mean,
                "difference_human_minus_ai": human_mean - ai_mean,
                "claim_frequency": claim_mean,
                "other_frequency": other_mean,
                "ratio_claim_to_other": (claim_mean / other_mean) if other_mean else None,
                "verdict": verdict,
            }
        )

    return rows


def write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "label",
                "title",
                "claim_label",
                "metric_name",
                "human_frequency",
                "ai_frequency",
                "difference_human_minus_ai",
                "claim_frequency",
                "other_frequency",
                "ratio_claim_to_other",
                "verdict",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_single_category(
    title: str,
    claim_label: str,
    human_mean: float,
    ai_mean: float,
    verdict: str,
    output_path: Path,
    value_fmt: str = "%.3f",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(LABELS)
    values = [human_mean, ai_mean]
    bars = ax.bar(labels, values, color=[LABEL_COLORS[label] for label in labels], width=0.55)
    ax.bar_label(bars, fmt=value_fmt, padding=2, fontsize=9)
    ax.set_ylabel("Mean value / share of comments")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)

    verdict_color = VERDICT_COLORS[verdict]
    ax.text(
        0.5,
        0.95,
        f"claim: {claim_label} | {verdict}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=verdict_color,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overview_grid(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 4, figsize=(24, 22))
    flattened_axes = list(axes.flat)

    for axis, row in zip(flattened_axes, rows):
        values = [float(row["human_frequency"]), float(row["ai_frequency"])]
        labels = list(LABELS)
        bars = axis.bar(labels, values, color=[LABEL_COLORS[label] for label in labels], width=0.55)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        axis.set_title(f"{row['category']}. {row['title']}", fontsize=10)
        axis.text(
            0.5,
            0.95,
            f"claim: {row['claim_label']} | {row['verdict']}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=VERDICT_COLORS[str(row["verdict"])],
        )
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelsize=8)
        axis.tick_params(axis="y", labelsize=8)

    for axis in flattened_axes[len(rows):]:
        axis.axis("off")

    fig.suptitle("Sentence and paragraph heuristic fact-checks by label", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and fact-check graphs for sentence and paragraph heuristics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "train.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--overview-output",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_factcheck_overview.png",
        help="Where to save the overview grid plot.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_factcheck_summary.csv",
        help="Where to save the CSV summary.",
    )
    parser.add_argument(
        "--individual-dir",
        type=Path,
        default=ARTIFACTS_DIR / "sentence_category_factcheck_individuals",
        help="Directory for per-category plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_counts, label_sums = analyze_dataset(args.input)
    rows = summarize_results(label_counts, label_sums)

    write_summary_csv(args.summary_csv, rows)
    plot_overview_grid(rows, args.overview_output)

    args.individual_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        file_name = f"{int(row['category']):02d}_{CATEGORIES[int(row['category']) - 1].slug}.png"
        plot_single_category(
            title=f"{row['category']}. {row['title']}",
            claim_label=str(row["claim_label"]),
            human_mean=float(row["human_frequency"]),
            ai_mean=float(row["ai_frequency"]),
            verdict=str(row["verdict"]),
            output_path=args.individual_dir / file_name,
        )

    print(f"Analyzed {sum(label_counts.values()):,} comments from {args.input}")
    for row in rows:
        print(
            f"{int(row['category']):02d}. {row['title']}: "
            f"human={float(row['human_frequency']):.4f}, ai={float(row['ai_frequency']):.4f}, "
            f"claim={row['claim_label']}, verdict={row['verdict']}"
        )
    print(f"Saved overview to {args.overview_output}")
    print(f"Saved summary to {args.summary_csv}")
    print(f"Saved individual plots to {args.individual_dir}")


if __name__ == "__main__":
    main()