from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

COMPONENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = COMPONENT_DIR.parent
ARTIFACTS_DIR = COMPONENT_DIR / "artifacts"
LABEL_TO_INT = {"HUMAN": 0, "AI": 1}
INT_TO_LABEL = {0: "HUMAN", 1: "AI"}
EPSILON = 1e-5

TREE_DIR = PROJECT_ROOT / "Tree_Based_Model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TREE_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_DIR))

from semantic_svm.embedding_extractor import EmbeddingExtractor, resolve_model_name  # noqa: E402
from sentence_category_factcheck import (
    CATEGORIES,
    compute_comment_metrics,
)  # noqa: E402


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def load_split(
    csv_path: Path,
    text_column: str = "comment",
    label_column: str = "label",
    require_labels: bool = True,
    max_rows: int | None = None,
) -> tuple[pd.Series, np.ndarray | None]:
    df = pd.read_csv(csv_path, nrows=max_rows)
    if text_column not in df.columns:
        raise ValueError(f"{csv_path} must contain a {text_column!r} column")

    comments = df[text_column].fillna("").astype(str)
    if label_column not in df.columns:
        if require_labels:
            raise ValueError(f"{csv_path} must contain a {label_column!r} column")
        return comments, None

    labels = df[label_column].fillna("").astype(str).str.strip().str.upper()
    unknown = sorted(set(labels) - set(LABEL_TO_INT))
    if unknown and require_labels:
        raise ValueError(f"Unknown labels in {csv_path}: {unknown}")

    if unknown:
        return comments, None
    return comments, labels.map(LABEL_TO_INT).to_numpy(dtype=np.int64)


def build_tf_logreg_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_df=0.95,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_df=0.95,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_tree_model(random_state: int) -> HistGradientBoostingClassifier:
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


def build_svm_model(random_state: int, calibration_cv: int, n_jobs: int):
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            LinearSVC(dual=False, max_iter=5000, random_state=random_state),
            method="sigmoid",
            cv=calibration_cv,
            n_jobs=n_jobs,
        ),
    )


def predict_ai_probability(model, values) -> np.ndarray:
    classes = list(model.classes_)
    if "AI" in classes:
        ai_index = classes.index("AI")
    else:
        ai_index = classes.index(LABEL_TO_INT["AI"])
    return model.predict_proba(values)[:, ai_index]


def compute_tree_features(comments: pd.Series) -> np.ndarray:
    feature_keys = [category.metric_key for category in CATEGORIES]
    rows: list[list[float]] = []
    for comment in comments:
        metrics = compute_comment_metrics(comment)
        rows.append([metrics[key] for key in feature_keys])
    return np.asarray(rows, dtype=np.float32)


def resolve_artifact_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    if not path.is_absolute():
        relative_to_manifest = manifest_path.parent / path
        if relative_to_manifest.exists():
            return relative_to_manifest

        relative_to_root = PROJECT_ROOT / path
        if relative_to_root.exists():
            return relative_to_root

    windows_path = PureWindowsPath(path_value)
    if windows_path.name:
        local_embedding_path = manifest_path.parent / "embeddings" / windows_path.name
        if local_embedding_path.exists():
            return local_embedding_path

    return path


def load_embeddings(
    manifest_path: Path,
    comments_by_split: dict[str, pd.Series],
    batch_size: int,
    device: str | None,
    model_cache_dir: Path | None,
) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings: dict[str, np.ndarray] = {}
    extractor: EmbeddingExtractor | None = None

    for split_name in ("train", "dev", "test"):
        cache_key = f"{split_name}_cache"
        cache_path_value = manifest.get(cache_key)
        if cache_path_value:
            cache_path = resolve_artifact_path(str(cache_path_value), manifest_path)
            if cache_path.exists():
                embeddings[split_name] = np.load(cache_path, mmap_mode="r")
                continue

            log(
                f"Embedding cache for {split_name} not found at {cache_path_value}; "
                "generating it from CSV comments"
            )

        if split_name not in comments_by_split:
            continue

        if extractor is None:
            model_name = resolve_model_name(
                str(manifest.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
            )
            log(f"Loading embedding model: {model_name}")
            extractor = EmbeddingExtractor(
                model_name=model_name,
                device=device,
                cache_folder=model_cache_dir,
            )

        embeddings[split_name] = extractor.encode(
            comments_by_split[split_name].tolist(),
            batch_size=batch_size,
            show_progress=True,
        )

    return embeddings


def inspect_tf_joblib(model_path: Path) -> dict[str, object]:
    if not model_path.exists():
        return {"exists": False}

    model = joblib.load(model_path)
    summary: dict[str, object] = {
        "exists": True,
        "type": type(model).__name__,
        "contains_precomputed_probabilities": False,
    }
    if hasattr(model, "named_steps"):
        summary["pipeline_steps"] = list(model.named_steps.keys())
        classifier = model.named_steps.get("classifier")
        if classifier is not None:
            summary["classes"] = list(classifier.classes_)
        features = model.named_steps.get("features")
        if features is not None:
            transformers = dict(features.transformer_list)
            if "word_tfidf" in transformers:
                summary["word_feature_count"] = int(
                    len(transformers["word_tfidf"].get_feature_names_out())
                )
            if "char_tfidf" in transformers:
                summary["char_feature_count"] = int(
                    len(transformers["char_tfidf"].get_feature_names_out())
                )
    return summary


def logit_features(probabilities: np.ndarray) -> tuple[np.ndarray, list[str]]:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    l_sem, l_tree, l_tf = logits[:, 0], logits[:, 1], logits[:, 2]
    matrix = np.column_stack(
        [
            l_sem,
            l_tree,
            l_tf,
            l_sem * l_tree,
            l_sem * l_tf,
            l_tree * l_tf,
            l_sem**2,
            l_tree**2,
            l_tf**2,
        ]
    )
    names = [
        "logit_semantic_svm",
        "logit_tree_based",
        "logit_tf_logreg",
        "logit_semantic_svm_x_tree_based",
        "logit_semantic_svm_x_tf_logreg",
        "logit_tree_based_x_tf_logreg",
        "logit_semantic_svm_sq",
        "logit_tree_based_sq",
        "logit_tf_logreg_sq",
    ]
    return matrix, names


def evaluate_probabilities(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def save_probability_csv(
    output_path: Path,
    probabilities: np.ndarray,
    meta_probabilities: np.ndarray | None,
    labels: np.ndarray | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(len(probabilities)),
            "p_semantic_svm_ai": probabilities[:, 0],
            "p_tree_based_ai": probabilities[:, 1],
            "p_tf_logreg_ai": probabilities[:, 2],
        }
    )
    if meta_probabilities is not None:
        frame["p_meta_ai"] = meta_probabilities
    if labels is not None:
        frame["label"] = [INT_TO_LABEL[int(label)] for label in labels]
    frame.to_csv(output_path, index=False)


def train_oof_base_probabilities(
    comments: pd.Series,
    labels: np.ndarray,
    train_embeddings: np.ndarray,
    tree_features: np.ndarray,
    folds: int,
    random_state: int,
    svm_calibration_cv: int,
    n_jobs: int,
) -> np.ndarray:
    oof_probabilities = np.zeros((len(labels), 3), dtype=np.float64)
    labels_as_text = pd.Series([INT_TO_LABEL[int(label)] for label in labels])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    for fold_index, (fit_indices, holdout_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        log(f"OOF fold {fold_index}/{folds}: fitting semantic SVM")
        svm_model = build_svm_model(
            random_state=random_state + fold_index,
            calibration_cv=svm_calibration_cv,
            n_jobs=n_jobs,
        )
        svm_model.fit(train_embeddings[fit_indices], labels[fit_indices])
        oof_probabilities[holdout_indices, 0] = predict_ai_probability(
            svm_model, train_embeddings[holdout_indices]
        )

        log(f"OOF fold {fold_index}/{folds}: fitting tree model")
        tree_model = build_tree_model(random_state=random_state + fold_index)
        tree_model.fit(tree_features[fit_indices], labels[fit_indices])
        oof_probabilities[holdout_indices, 1] = predict_ai_probability(
            tree_model, tree_features[holdout_indices]
        )

        log(f"OOF fold {fold_index}/{folds}: fitting TF-IDF logistic regression")
        tf_model = build_tf_logreg_model()
        tf_model.fit(comments.iloc[fit_indices], labels_as_text.iloc[fit_indices])
        oof_probabilities[holdout_indices, 2] = predict_ai_probability(
            tf_model, comments.iloc[holdout_indices]
        )

    return oof_probabilities


def train_final_base_probabilities(
    comments_train: pd.Series,
    labels_train: np.ndarray,
    comments_by_split: dict[str, pd.Series],
    embeddings_by_split: dict[str, np.ndarray],
    tree_features_by_split: dict[str, np.ndarray],
    svm_model_path: Path,
    output_dir: Path,
    random_state: int,
) -> dict[str, np.ndarray]:
    log("Loading full semantic SVM model")
    svm_model = joblib.load(svm_model_path)

    log("Fitting final tree model on full train")
    tree_model = build_tree_model(random_state=random_state)
    tree_model.fit(tree_features_by_split["train"], labels_train)
    joblib.dump(tree_model, output_dir / "tree_based_full_train.joblib")

    log("Fitting final TF-IDF logistic regression on full train")
    labels_as_text = pd.Series([INT_TO_LABEL[int(label)] for label in labels_train])
    tf_model = build_tf_logreg_model()
    tf_model.fit(comments_train, labels_as_text)
    joblib.dump(tf_model, output_dir / "tf_logreg_full_train.joblib")

    probabilities: dict[str, np.ndarray] = {}
    for split_name, comments in comments_by_split.items():
        log(f"Predicting final base probabilities for {split_name}")
        split_probabilities = np.column_stack(
            [
                predict_ai_probability(svm_model, embeddings_by_split[split_name]),
                predict_ai_probability(tree_model, tree_features_by_split[split_name]),
                predict_ai_probability(tf_model, comments),
            ]
        )
        probabilities[split_name] = split_probabilities

    return probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the three-model stacking meta-model."
    )
    parser.add_argument(
        "--train-csv", type=Path, default=PROJECT_ROOT / "data" / "train.csv"
    )
    parser.add_argument(
        "--dev-csv", type=Path, default=PROJECT_ROOT / "data" / "dev.csv"
    )
    parser.add_argument(
        "--test-csv", type=Path, default=PROJECT_ROOT / "data" / "test.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--svm-model",
        type=Path,
        default=PROJECT_ROOT / "semantic_svm" / "artifacts" / "semantic_svm.joblib",
    )
    parser.add_argument(
        "--svm-embedding-manifest",
        type=Path,
        default=PROJECT_ROOT / "semantic_svm" / "artifacts" / "embedding_manifest.json",
    )
    parser.add_argument(
        "--existing-tf-model",
        type=Path,
        default=PROJECT_ROOT / "tf-logreg" / "reddit_ai_detector.joblib",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--svm-calibration-cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-device", type=str, default=None)
    parser.add_argument("--embedding-model-cache-dir", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-dev-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading train/dev/test CSV files")
    train_comments, train_labels = load_split(
        args.train_csv, max_rows=args.max_train_rows
    )
    dev_comments, dev_labels = load_split(args.dev_csv, max_rows=args.max_dev_rows)
    test_comments, test_labels = load_split(
        args.test_csv,
        require_labels=False,
        max_rows=args.max_test_rows,
    )
    if train_labels is None or dev_labels is None:
        raise ValueError("Train and dev labels are required")

    log("Inspecting existing tf-logreg joblib")
    tf_joblib_summary = inspect_tf_joblib(args.existing_tf_model)
    (args.output_dir / "tf_logreg_existing_joblib_summary.json").write_text(
        json.dumps(tf_joblib_summary, indent=2), encoding="utf-8"
    )

    comments_by_split = {
        "train": train_comments,
        "dev": dev_comments,
        "test": test_comments,
    }

    log("Loading cached semantic SVM embeddings")
    embeddings = load_embeddings(
        args.svm_embedding_manifest,
        comments_by_split=comments_by_split,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
        model_cache_dir=args.embedding_model_cache_dir,
    )
    if args.max_train_rows is not None:
        embeddings["train"] = embeddings["train"][: args.max_train_rows]
    if args.max_dev_rows is not None:
        embeddings["dev"] = embeddings["dev"][: args.max_dev_rows]
    if args.max_test_rows is not None:
        embeddings["test"] = embeddings["test"][: args.max_test_rows]

    expected_lengths = {
        "train": len(train_comments),
        "dev": len(dev_comments),
        "test": len(test_comments),
    }
    for split_name, expected_length in expected_lengths.items():
        if split_name not in embeddings:
            raise ValueError(
                f"Missing {split_name} embeddings in {args.svm_embedding_manifest}"
            )
        if len(embeddings[split_name]) != expected_length:
            raise ValueError(
                f"{split_name} embeddings have {len(embeddings[split_name])} rows, "
                f"but CSV has {expected_length} rows"
            )

    log("Computing tree-based structural features")
    tree_features_by_split = {
        split_name: compute_tree_features(comments)
        for split_name, comments in comments_by_split.items()
    }

    log("Generating OOF base probabilities for train")
    train_oof_probabilities = train_oof_base_probabilities(
        comments=train_comments,
        labels=train_labels,
        train_embeddings=embeddings["train"],
        tree_features=tree_features_by_split["train"],
        folds=args.folds,
        random_state=args.random_state,
        svm_calibration_cv=args.svm_calibration_cv,
        n_jobs=args.n_jobs,
    )

    log("Training final base models and predicting train/dev/test probabilities")
    final_base_probabilities = train_final_base_probabilities(
        comments_train=train_comments,
        labels_train=train_labels,
        comments_by_split=comments_by_split,
        embeddings_by_split=embeddings,
        tree_features_by_split=tree_features_by_split,
        svm_model_path=args.svm_model,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )

    log("Training logistic stacking meta-model")
    meta_train_features, meta_feature_names = logit_features(train_oof_probabilities)
    meta_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    meta_model.fit(meta_train_features, train_labels)

    meta_train_probabilities = predict_ai_probability(meta_model, meta_train_features)
    meta_probabilities_by_split: dict[str, np.ndarray] = {
        "train_oof": meta_train_probabilities,
    }
    for split_name, base_probabilities in final_base_probabilities.items():
        split_features, _ = logit_features(base_probabilities)
        meta_probabilities_by_split[split_name] = predict_ai_probability(
            meta_model, split_features
        )

    log("Saving probability CSV files")
    save_probability_csv(
        args.output_dir / "train_oof_probabilities.csv",
        train_oof_probabilities,
        meta_probabilities_by_split["train_oof"],
        train_labels,
    )
    save_probability_csv(
        args.output_dir / "train_full_base_probabilities.csv",
        final_base_probabilities["train"],
        meta_probabilities_by_split["train"],
        train_labels,
    )
    save_probability_csv(
        args.output_dir / "dev_probabilities.csv",
        final_base_probabilities["dev"],
        meta_probabilities_by_split["dev"],
        dev_labels,
    )
    save_probability_csv(
        args.output_dir / "test_probabilities.csv",
        final_base_probabilities["test"],
        meta_probabilities_by_split["test"],
        test_labels,
    )

    log("Saving meta-model and metrics")
    joblib.dump(
        {
            "model": meta_model,
            "feature_names": meta_feature_names,
            "base_probability_columns": [
                "p_semantic_svm_ai",
                "p_tree_based_ai",
                "p_tf_logreg_ai",
            ],
            "epsilon": EPSILON,
        },
        args.output_dir / "stacking_logreg_meta_model.joblib",
    )

    metrics: dict[str, object] = {
        "train_rows": int(len(train_labels)),
        "dev_rows": int(len(dev_labels)),
        "test_rows": int(len(test_comments)),
        "folds": int(args.folds),
        "base_models": {
            "train_oof": {
                "semantic_svm": evaluate_probabilities(
                    train_labels, train_oof_probabilities[:, 0]
                ),
                "tree_based": evaluate_probabilities(
                    train_labels, train_oof_probabilities[:, 1]
                ),
                "tf_logreg": evaluate_probabilities(
                    train_labels, train_oof_probabilities[:, 2]
                ),
            },
            "dev": {
                "semantic_svm": evaluate_probabilities(
                    dev_labels, final_base_probabilities["dev"][:, 0]
                ),
                "tree_based": evaluate_probabilities(
                    dev_labels, final_base_probabilities["dev"][:, 1]
                ),
                "tf_logreg": evaluate_probabilities(
                    dev_labels, final_base_probabilities["dev"][:, 2]
                ),
            },
        },
        "meta_model": {
            "train_oof": evaluate_probabilities(
                train_labels, meta_probabilities_by_split["train_oof"]
            ),
            "dev": evaluate_probabilities(
                dev_labels, meta_probabilities_by_split["dev"]
            ),
        },
    }
    if test_labels is not None:
        metrics["meta_model"]["test"] = evaluate_probabilities(
            test_labels, meta_probabilities_by_split["test"]
        )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "train_csv": str(args.train_csv),
                "dev_csv": str(args.dev_csv),
                "test_csv": str(args.test_csv),
                "svm_model": str(args.svm_model),
                "svm_embedding_manifest": str(args.svm_embedding_manifest),
                "existing_tf_model_summary": str(
                    args.output_dir / "tf_logreg_existing_joblib_summary.json"
                ),
                "meta_model": str(
                    args.output_dir / "stacking_logreg_meta_model.joblib"
                ),
                "probability_files": [
                    str(args.output_dir / "train_oof_probabilities.csv"),
                    str(args.output_dir / "train_full_base_probabilities.csv"),
                    str(args.output_dir / "dev_probabilities.csv"),
                    str(args.output_dir / "test_probabilities.csv"),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics["meta_model"], indent=2))
    log(f"Saved meta-model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
