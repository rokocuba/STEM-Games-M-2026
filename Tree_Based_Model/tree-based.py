"""Train and evaluate a tree-based model on the sentence heuristics.

The model uses the 20 metrics defined in sentence_category_factcheck.py,
trains on data/train.csv, and evaluates on data/dev.csv.
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from sentence_category_factcheck import CATEGORIES, compute_comment_metrics


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LABEL_TO_INT = {"HUMAN": 0, "AI": 1}
TARGET_NAMES = ["HUMAN", "AI"]


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    feature_keys = [category.metric_key for category in CATEGORIES]
    feature_rows: list[list[float]] = []
    labels: list[int] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"comment", "label"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain comment and label columns")

        for row in reader:
            label = (row.get("label", "") or "").strip().upper()
            if label not in LABEL_TO_INT:
                continue

            metrics = compute_comment_metrics(row.get("comment", "") or "")
            feature_rows.append([metrics[key] for key in feature_keys])
            labels.append(LABEL_TO_INT[label])

    if not feature_rows:
        raise ValueError(f"No labeled rows found in {csv_path}")

    return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def build_boosted_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.06,
        max_iter=300,
        min_samples_leaf=15,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )


def build_decision_tree(
    random_state: int = 42,
    max_depth: int = 10,
    min_samples_leaf: int = 20,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )


def plot_feature_importances(output_path: Path, feature_names: list[str], importances: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(importances)
    sorted_features = [feature_names[index] for index in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_features, sorted_importances, color="#4c72b0")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Feature importance")
    ax.set_title("Feature importances")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def collect_decision_path_steps(
    model: DecisionTreeClassifier,
    sample: np.ndarray,
) -> tuple[list[dict[str, object]], int]:
    tree = model.tree_
    node_indicator = model.decision_path(sample.reshape(1, -1))
    leaf_id = int(model.apply(sample.reshape(1, -1))[0])
    steps: list[dict[str, object]] = []

    for node_id in node_indicator.indices:
        feature_index = int(tree.feature[node_id])
        if feature_index == -2:
            steps.append(
                {
                    "node_id": int(node_id),
                    "is_leaf": True,
                }
            )
            continue

        threshold = float(tree.threshold[node_id])
        value = float(sample[feature_index])
        direction = "left" if value <= threshold else "right"
        steps.append(
            {
                "node_id": int(node_id),
                "is_leaf": False,
                "feature_index": feature_index,
                "threshold": threshold,
                "value": value,
                "direction": direction,
            }
        )

    return steps, leaf_id


def describe_decision_path(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    sample: np.ndarray,
    case_index: int,
    true_label: int,
    predicted_label: int,
    predicted_probability: float,
) -> None:
    steps, leaf_id = collect_decision_path_steps(model, sample)

    print(
        f"Case {case_index + 1}: true={TARGET_NAMES[true_label]} "
        f"predicted={TARGET_NAMES[predicted_label]} prob_AI={predicted_probability:.4f}"
    )
    for step in steps:
        if step["is_leaf"]:
            values = model.tree_.value[int(step["node_id"])][0]
            print(
                f"  Leaf {int(step['node_id'])}: class_counts="
                f"{{HUMAN={values[0]:.3f}, AI={values[1]:.3f}}}"
            )
            continue

        feature_index = int(step["feature_index"])
        threshold = float(step["threshold"])
        value = float(step["value"])
        direction = str(step["direction"])
        operator = "<=" if direction == "left" else ">"
        print(
            f"  Node {int(step['node_id'])}: {feature_names[feature_index]} = {value:.4f} "
            f"{operator} {threshold:.4f} -> {direction}"
        )

    print(f"  Reached leaf {leaf_id}")


def save_figure_pair(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    companion_path = output_path.with_suffix(".svg") if output_path.suffix.lower() != ".svg" else output_path.with_suffix(".png")
    fig.savefig(companion_path, bbox_inches="tight")


def render_decision_path_image(
    output_path: Path,
    model: DecisionTreeClassifier,
    feature_names: list[str],
    sample: np.ndarray,
    sample_index: int,
    source_name: str,
    true_label: int,
    predicted_label: int,
    predicted_probability: float,
) -> None:
    steps, leaf_id = collect_decision_path_steps(model, sample)
    leaf_proba = model.predict_proba(sample.reshape(1, -1))[0]

    fig_height = max(6.5, 1.25 * len(steps) + 2.6)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps) + 2.8)

    title_y = len(steps) + 2.1
    subtitle_y = len(steps) + 1.55
    ax.text(
        0.5,
        title_y,
        "Decision Path for One Example",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#e2e8f0", edgecolor="#cbd5e1", linewidth=1.5),
    )
    ax.text(
        0.5,
        subtitle_y,
        (
            f"Case {sample_index + 1} from {source_name} | true={TARGET_NAMES[true_label]} | "
            f"predicted={TARGET_NAMES[predicted_label]} | P(AI)={predicted_probability:.3f} | "
            f"visited nodes={len(steps)}"
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    box_width = 0.84
    box_height = 0.86
    center_x = 0.5
    y_top = len(steps) + 0.9

    for index, step in enumerate(steps):
        node_y = y_top - index * 1.05

        if step["is_leaf"]:
            correct = predicted_label == true_label
            edge_color = "#16a34a" if correct else "#dc2626"
            face_color = "#ecfdf5" if correct else "#fef2f2"
            box_text = "\n".join(
                [
                    f"Leaf {int(step['node_id'])}",
                    f"P(HUMAN)={leaf_proba[0]:.3f}",
                    f"P(AI)={leaf_proba[1]:.3f}",
                    f"predicted {TARGET_NAMES[predicted_label]}",
                ]
            )
            ax.text(
                center_x,
                node_y,
                box_text,
                ha="center",
                va="center",
                fontsize=12,
                color="#0f172a",
                linespacing=1.25,
                bbox=dict(
                    boxstyle="round,pad=0.65",
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=2.6,
                ),
            )
        else:
            feature_index = int(step["feature_index"])
            direction = str(step["direction"])
            edge_color = "#2563eb" if direction == "left" else "#ea580c"
            face_color = "#ffffff"
            operator = "<=" if direction == "left" else ">"
            condition_text = textwrap.fill(feature_names[feature_index], width=28)
            box_text = "\n".join(
                [
                    f"Node {int(step['node_id'])}",
                    condition_text,
                    f"{float(step['value']):.3f} {operator} {float(step['threshold']):.3f}",
                    f"take {direction.upper()}",
                ]
            )
            ax.text(
                center_x,
                node_y,
                box_text,
                ha="center",
                va="center",
                fontsize=12,
                color="#0f172a",
                linespacing=1.2,
                bbox=dict(
                    boxstyle="round,pad=0.65",
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=2.4,
                ),
            )

        if index < len(steps) - 1:
            next_direction = str(step["direction"]) if not step["is_leaf"] else "right"
            arrow_color = "#2563eb" if next_direction == "left" else "#ea580c"
            ax.annotate(
                "",
                xy=(center_x, node_y - box_height / 2 + 0.02),
                xytext=(center_x, node_y - 1.05 + box_height / 2 - 0.02),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=arrow_color,
                    linewidth=2.4,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=18,
                ),
            )

    ax.text(
        0.5,
        0.35,
        "Blue = left branch, orange = right branch, green = correct leaf",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )

    fig.tight_layout()
    save_figure_pair(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a tree-based model.")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "train.csv", help="Training CSV path.")
    parser.add_argument("--dev", type=Path, default=PROJECT_ROOT / "data" / "dev.csv", help="Development CSV path.")
    parser.add_argument(
        "--model",
        choices=["boosted", "decision-tree"],
        default="boosted",
        help="Which tree model to train.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=10,
        help="Maximum depth for the single decision tree model.",
    )
    parser.add_argument(
        "--tree-min-samples-leaf",
        type=int,
        default=20,
        help="Minimum samples per leaf for the single decision tree model.",
    )
    parser.add_argument(
        "--importance-sample-size",
        type=int,
        default=4000,
        help="Number of dev samples to use when computing permutation importance for the boosted model.",
    )
    parser.add_argument(
        "--explain-count",
        type=int,
        default=0,
        help="Print the full decision path for the first N examples from the chosen source dataset.",
    )
    parser.add_argument(
        "--explain-source",
        choices=["dev", "train"],
        default="dev",
        help="Which dataset to use for the decision-path explanation.",
    )
    parser.add_argument(
        "--path-image-output",
        type=Path,
        default=None,
        help="Optional image path for a single-example decision-path diagram.",
    )
    parser.add_argument(
        "--path-case-index",
        type=int,
        default=0,
        help="0-based example index to render when creating a decision-path image.",
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=ARTIFACTS_DIR / "tree_based_feature_importances.png",
        help="Where to save the feature importance plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_train, y_train = load_dataset(args.train)
    x_dev, y_dev = load_dataset(args.dev)

    if args.model == "decision-tree":
        model = build_decision_tree(
            max_depth=args.tree_max_depth,
            min_samples_leaf=args.tree_min_samples_leaf,
        )
    else:
        model = build_boosted_model()

    model.fit(x_train, y_train)

    predicted_labels = model.predict(x_dev)
    predicted_probabilities = model.predict_proba(x_dev)[:, 1]

    accuracy = accuracy_score(y_dev, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(y_dev, predicted_labels)
    precision = precision_score(y_dev, predicted_labels)
    recall = recall_score(y_dev, predicted_labels)
    f1 = f1_score(y_dev, predicted_labels)
    auc = roc_auc_score(y_dev, predicted_probabilities)
    matrix = confusion_matrix(y_dev, predicted_labels)

    feature_names = [category.title for category in CATEGORIES]

    if args.model == "decision-tree":
        importances = model.feature_importances_
    else:
        rng = np.random.default_rng(42)
        sample_size = min(args.importance_sample_size, len(x_dev))
        sample_indices = rng.choice(len(x_dev), size=sample_size, replace=False)
        sample_x = x_dev[sample_indices]
        sample_y = y_dev[sample_indices]

        importance_result = permutation_importance(
            model,
            sample_x,
            sample_y,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring="accuracy",
        )
        importances = importance_result.importances_mean

    top_indices = np.argsort(importances)[::-1]

    plot_feature_importances(args.feature_output, feature_names, importances)

    if args.path_image_output is not None:
        if args.model != "decision-tree":
            print("Decision-path images require --model decision-tree; rerun with that mode to render one.")
        else:
            image_x, image_y = (x_dev, y_dev) if args.explain_source == "dev" else (x_train, y_train)
            image_index = min(max(args.path_case_index, 0), len(image_x) - 1)
            image_sample = image_x[image_index]
            image_true_label = int(image_y[image_index])
            image_predicted_label = int(model.predict(image_sample.reshape(1, -1))[0])
            image_predicted_probability = float(model.predict_proba(image_sample.reshape(1, -1))[0, 1])

            render_decision_path_image(
                output_path=args.path_image_output,
                model=model,
                feature_names=feature_names,
                sample=image_sample,
                sample_index=image_index,
                source_name=args.explain_source,
                true_label=image_true_label,
                predicted_label=image_predicted_label,
                predicted_probability=image_predicted_probability,
            )
            print(f"Saved decision-path image to {args.path_image_output}")

    if args.explain_count > 0:
        if args.model != "decision-tree":
            print("Exact tree walks require --model decision-tree; rerun with that mode to inspect paths.")
        else:
            explain_x, explain_y = (x_dev, y_dev) if args.explain_source == "dev" else (x_train, y_train)
            explain_count = min(args.explain_count, len(explain_x))
            explain_predicted_labels = model.predict(explain_x[:explain_count])
            explain_predicted_probabilities = model.predict_proba(explain_x[:explain_count])[:, 1]

            print(
                f"Full tree walk for the first {explain_count} {args.explain_source} cases "
                f"using the decision-tree model:"
            )
            for index in range(explain_count):
                describe_decision_path(
                    model=model,
                    feature_names=feature_names,
                    sample=explain_x[index],
                    case_index=index,
                    true_label=int(explain_y[index]),
                    predicted_label=int(explain_predicted_labels[index]),
                    predicted_probability=float(explain_predicted_probabilities[index]),
                )

    print(f"Trained on {len(y_train):,} comments from {args.train}")
    print(f"Tested on {len(y_dev):,} comments from {args.dev}")
    print(f"Model type: {args.model}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)
    print("Classification report:")
    print(classification_report(y_dev, predicted_labels, target_names=TARGET_NAMES, digits=4))
    print("Top feature importances:")
    for index in top_indices[:10]:
        print(f"  {feature_names[index]}: {importances[index]:.4f}")
    print(f"Saved feature importance plot to {args.feature_output}")


if __name__ == "__main__":
    main()