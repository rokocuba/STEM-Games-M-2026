"""Analyze text signals in data/train.csv.

The script produces three grouped bar charts:
1. punctuation hyper-correctness
2. macro-structural sentence patterns
3. expected punctuation after specific words and syntagms

Each chart compares HUMAN and AI comments, writes a summary CSV, and saves
the plot to Tree_Based_Model/artifacts/.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


FeatureSpec = tuple[str, str, Callable[[str], int]]

LABEL_ORDER = ("HUMAN", "AI")
LABEL_COLORS = {
    "HUMAN": "#1f77b4",
    "AI": "#ff7f0e",
}

CONTRACTION_RE = re.compile(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b", re.IGNORECASE)
OXFORD_COMMA_RE = re.compile(
    r"\b[\w'.-]+(?:\s+[\w'.-]+){0,2},\s+[\w'.-]+(?:\s+[\w'.-]+){0,2},\s+(?:and|or)\s+[\w'.-]+(?:\s+[\w'.-]+){0,2}",
    re.IGNORECASE,
)
BALANCED_COUNTER_WEIGHT_PATTERNS = (
    re.compile(r"\bnot only\b.*?\bbut also\b", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"\b(?:it['’]s|it is)\s+not\s+about\b.*?\b(?:it['’]s|it is)\s+about\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
RULE_OF_THREE_RE = re.compile(
    r"(?:^|[.!?\n]\s*)(?:[^.!?\n]+?,\s*){2}(?:and|or)\s+[^.!?\n]+",
    re.IGNORECASE,
)
LIST_DEFAULT_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
WRAP_UP_RE = re.compile(
    r"(?:^|[.!?\n]\s*)(?:ultimately|in conclusion|at the end of the day)\b",
    re.IGNORECASE,
)
EMOTION_INTERJECTION_RE = re.compile(
    r"\b(?:oh\s+no|oh\s+boy|hooray|wow|alas|yay|awesome|amazing|great|nice)\b\s*!",
    re.IGNORECASE,
)
TRANSITION_PHRASE_RE = re.compile(
    r"\b(?:on one hand|for example|furthermore|however|moreover|in addition|therefore|thus|instead|still)\b\s*,",
    re.IGNORECASE,
)
POLITE_REFUSAL_RE = re.compile(
    r"\b(?:that['’]s a great point|that['’]s fair|i understand your perspective|i see what you mean|i get your point|fair enough|that said|i['’]m not saying)\b\s*[,—-]\s*(?:but|however|though|yet)\b",
    re.IGNORECASE,
)
GREETING_RE = re.compile(
    r"\b(?:thanks for asking|thanks for the question|thank you for asking|thanks|hey there|hello|hi|hey|appreciate it)\b\s*[,:\-—]",
    re.IGNORECASE,
)
RHETORICAL_PIVOT_RE = re.compile(
    r"\b(?:here['’]s the thing|here['’]s why|look|the thing is|point is|bottom line)\b\s*:",
    re.IGNORECASE,
)


def ordered_labels(label_counts: Counter[str]) -> list[str]:
    """Keep the common HUMAN vs AI ordering when both labels are present."""

    labels = [label for label in LABEL_ORDER if label in label_counts]
    labels.extend(sorted(label for label in label_counts if label not in LABEL_ORDER))
    return labels


def count_balanced_counter_weight(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in BALANCED_COUNTER_WEIGHT_PATTERNS)


def count_rule_of_three(text: str) -> int:
    return len(RULE_OF_THREE_RE.findall(text))


def count_list_default(text: str) -> int:
    return len(LIST_DEFAULT_RE.findall(text))


def count_wrap_up(text: str) -> int:
    return len(WRAP_UP_RE.findall(text))


def count_emotion_interjection(text: str) -> int:
    return len(EMOTION_INTERJECTION_RE.findall(text))


def count_transition_phrase(text: str) -> int:
    return len(TRANSITION_PHRASE_RE.findall(text))


def count_polite_refusal(text: str) -> int:
    return len(POLITE_REFUSAL_RE.findall(text))


def count_greeting_punctuation(text: str) -> int:
    return len(GREETING_RE.findall(text))


def count_rhetorical_pivot(text: str) -> int:
    return len(RHETORICAL_PIVOT_RE.findall(text))


PUNCTUATION_FEATURES: tuple[FeatureSpec, ...] = (
    ("em_dash", "Em dash (—)", lambda text: text.count("—")),
    ("semicolon", "Semicolon", lambda text: text.count(";")),
    ("oxford_comma", "Oxford comma", lambda text: len(OXFORD_COMMA_RE.findall(text))),
    ("apostrophe", "Apostrophes", lambda text: len(CONTRACTION_RE.findall(text))),
)

MACRO_FEATURES: tuple[FeatureSpec, ...] = (
    ("balanced_counter_weight", "Balanced counter-weight", count_balanced_counter_weight),
    ("rule_of_three", "Rule of three", count_rule_of_three),
    ("list_default", "Bulleted/numbered list", count_list_default),
    ("wrap_up", "Wrap-up phrase", count_wrap_up),
)

SYNTAGM_FEATURES: tuple[FeatureSpec, ...] = (
    ("emotion_interjection", "Emotion interjections (!)", count_emotion_interjection),
    ("transition_phrase", "Transition phrases (,)", count_transition_phrase),
    ("polite_refusal", "Polite refusals (,-)", count_polite_refusal),
    ("greeting_punctuation", "Greetings / direct address (,:)", count_greeting_punctuation),
    ("rhetorical_pivot", "Rhetorical pivots (:)", count_rhetorical_pivot),
)


def count_features(text: str, feature_specs: Sequence[FeatureSpec]) -> dict[str, int]:
    return {feature_key: extractor(text) for feature_key, _, extractor in feature_specs}


def analyze_dataset(
    csv_path: Path,
    feature_specs: Sequence[FeatureSpec],
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    label_counts: Counter[str] = Counter()
    feature_totals: dict[str, Counter[str]] = defaultdict(Counter)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"comment", "label"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain comment and label columns")

        for row in reader:
            comment = row.get("comment", "") or ""
            label = (row.get("label", "") or "").strip().upper()
            if not label:
                continue

            label_counts[label] += 1
            for feature, count in count_features(comment, feature_specs).items():
                feature_totals[label][feature] += count

    return label_counts, feature_totals


def write_summary_csv(
    output_path: Path,
    label_counts: Counter[str],
    feature_totals: dict[str, Counter[str]],
    feature_specs: Sequence[FeatureSpec],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for label in ordered_labels(label_counts):
        comment_count = label_counts[label]
        for feature, display_name, _ in feature_specs:
            total = feature_totals[label][feature]
            rows.append(
                {
                    "label": label,
                    "feature": display_name,
                    "total_count": total,
                    "comments": comment_count,
                    "mean_per_comment": total / comment_count if comment_count else 0.0,
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "feature", "total_count", "comments", "mean_per_comment"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_feature_summary(
    output_path: Path,
    label_counts: Counter[str],
    feature_totals: dict[str, Counter[str]],
    feature_specs: Sequence[FeatureSpec],
    title: str,
    bar_label_fmt: str = "%.2f",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ordered_labels(label_counts)
    feature_keys = [feature for feature, _, _ in feature_specs]
    feature_names = [display_name for _, display_name, _ in feature_specs]
    positions = list(range(len(feature_keys)))
    bar_width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for index, label in enumerate(labels):
        offset = (index - (len(labels) - 1) / 2) * bar_width
        values = [
            feature_totals[label][feature] / label_counts[label]
            for feature in feature_keys
        ]
        bars = ax.bar(
            [position + offset for position in positions],
            values,
            width=bar_width,
            label=label,
            color=LABEL_COLORS.get(label),
        )
        ax.bar_label(bars, fmt=bar_label_fmt, padding=2, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(feature_names, rotation=15, ha="right")
    ax.set_ylabel("Average occurrences per comment")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(
    csv_path: Path,
    label_counts: Counter[str],
    feature_totals: dict[str, Counter[str]],
    feature_specs: Sequence[FeatureSpec],
) -> None:
    total_comments = sum(label_counts.values())
    print(f"Analyzed {total_comments:,} comments from {csv_path}")

    for label in ordered_labels(label_counts):
        comment_count = label_counts[label]
        print(f"{label}: {comment_count:,} comments")
        for feature, display_name, _ in feature_specs:
            total = feature_totals[label][feature]
            mean = total / comment_count if comment_count else 0.0
            print(f"  {display_name}: {total:,} total, {mean:.4f} per comment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze punctuation hyper-correctness signals in train.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "train.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--punctuation-output",
        type=Path,
        default=ARTIFACTS_DIR / "punctuation_hyper_correctness_by_label.png",
        help="Where to save the punctuation plot.",
    )
    parser.add_argument(
        "--punctuation-summary-csv",
        type=Path,
        default=ARTIFACTS_DIR / "punctuation_hyper_correctness_summary.csv",
        help="Where to save the punctuation tabular summary.",
    )
    parser.add_argument(
        "--macro-output",
        type=Path,
        default=ARTIFACTS_DIR / "macro_structural_rules_by_label.png",
        help="Where to save the macro-structure plot.",
    )
    parser.add_argument(
        "--macro-summary-csv",
        type=Path,
        default=ARTIFACTS_DIR / "macro_structural_rules_summary.csv",
        help="Where to save the macro-structure tabular summary.",
    )
    parser.add_argument(
        "--syntagm-output",
        type=Path,
        default=ARTIFACTS_DIR / "expected_characters_after_syntagms_by_label.png",
        help="Where to save the expected-character plot.",
    )
    parser.add_argument(
        "--syntagm-summary-csv",
        type=Path,
        default=ARTIFACTS_DIR / "expected_characters_after_syntagms_summary.csv",
        help="Where to save the expected-character tabular summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    punct_label_counts, punct_feature_totals = analyze_dataset(args.input, PUNCTUATION_FEATURES)
    write_summary_csv(
        args.punctuation_summary_csv,
        punct_label_counts,
        punct_feature_totals,
        PUNCTUATION_FEATURES,
    )
    plot_feature_summary(
        args.punctuation_output,
        punct_label_counts,
        punct_feature_totals,
        PUNCTUATION_FEATURES,
        "Punctuation hyper-correctness by label",
    )
    print_summary(args.input, punct_label_counts, punct_feature_totals, PUNCTUATION_FEATURES)
    print(f"Saved plot to {args.punctuation_output}")
    print(f"Saved summary to {args.punctuation_summary_csv}")

    macro_label_counts, macro_feature_totals = analyze_dataset(args.input, MACRO_FEATURES)
    write_summary_csv(
        args.macro_summary_csv,
        macro_label_counts,
        macro_feature_totals,
        MACRO_FEATURES,
    )
    plot_feature_summary(
        args.macro_output,
        macro_label_counts,
        macro_feature_totals,
        MACRO_FEATURES,
        "Macro-structural rules by label",
        bar_label_fmt="%.3f",
    )
    print_summary(args.input, macro_label_counts, macro_feature_totals, MACRO_FEATURES)
    print(f"Saved plot to {args.macro_output}")
    print(f"Saved summary to {args.macro_summary_csv}")

    syntagm_label_counts, syntagm_feature_totals = analyze_dataset(args.input, SYNTAGM_FEATURES)
    write_summary_csv(
        args.syntagm_summary_csv,
        syntagm_label_counts,
        syntagm_feature_totals,
        SYNTAGM_FEATURES,
    )
    plot_feature_summary(
        args.syntagm_output,
        syntagm_label_counts,
        syntagm_feature_totals,
        SYNTAGM_FEATURES,
        "Expected characters after specific words/syntagms by label",
        bar_label_fmt="%.3f",
    )
    print_summary(args.input, syntagm_label_counts, syntagm_feature_totals, SYNTAGM_FEATURES)
    print(f"Saved plot to {args.syntagm_output}")
    print(f"Saved summary to {args.syntagm_summary_csv}")


if __name__ == "__main__":
    main()
