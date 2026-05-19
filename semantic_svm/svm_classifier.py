from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

try:
    from .embedding_extractor import (
        DEFAULT_MODEL_NAME,
        EmbeddingExtractor,
        count_rows,
        resolve_model_name,
    )
except ImportError:
    from embedding_extractor import (
        DEFAULT_MODEL_NAME,
        EmbeddingExtractor,
        count_rows,
        resolve_model_name,
    )


COMPONENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = COMPONENT_DIR / "artifacts"
LABEL_TO_ID = {"HUMAN": 0, "AI": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
AI_CLASS_ID = LABEL_TO_ID["AI"]


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


@contextmanager
def progress_step(label: str, enabled: bool = True):
    log(f"Starting: {label}")
    start = perf_counter()
    with tqdm(
        total=1, desc=label, unit="step", dynamic_ncols=True, disable=not enabled
    ) as bar:
        try:
            yield
        except Exception:
            log(f"Failed: {label} after {format_seconds(perf_counter() - start)}")
            raise
        else:
            bar.update(1)
            log(f"Finished: {label} in {format_seconds(perf_counter() - start)}")


def load_labels(
    csv_path: str | Path,
    label_column: str = "label",
    chunksize: int = 4096,
    max_rows: int | None = None,
    progress: bool = True,
) -> np.ndarray:
    csv_path = Path(csv_path)
    rows_left = max_rows
    label_chunks: list[pd.Series] = []

    with tqdm(
        desc=f"Loading labels from {csv_path.name}",
        unit="rows",
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        for chunk in pd.read_csv(csv_path, usecols=[label_column], chunksize=chunksize):
            if rows_left is not None:
                if rows_left <= 0:
                    break
                chunk = chunk.head(rows_left)
                rows_left -= len(chunk)

            label_chunks.append(chunk[label_column])
            bar.update(len(chunk))

    if not label_chunks:
        raise ValueError(f"No labels found in {csv_path}")

    labels = pd.concat(label_chunks, ignore_index=True)
    normalized = labels.fillna("").astype(str).str.strip().str.upper()
    unknown = sorted(set(normalized) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"Unknown labels in {csv_path}: {unknown}")
    return normalized.map(LABEL_TO_ID).to_numpy(dtype=np.int64)


def load_text_sample(
    csv_path: str | Path,
    text_column: str,
    sample_rows: int,
) -> list[str]:
    if sample_rows <= 0:
        return []
    sample = pd.read_csv(csv_path, usecols=[text_column], nrows=sample_rows)
    return sample[text_column].fillna("").astype(str).tolist()


def estimate_embedding_runtime(
    extractor: EmbeddingExtractor,
    csv_path: str | Path,
    text_column: str,
    total_rows: int,
    batch_size: int,
    sample_rows: int,
) -> tuple[float, float] | None:
    texts = load_text_sample(csv_path, text_column, min(sample_rows, total_rows))
    if not texts:
        return None

    probe_batch_size = min(batch_size, len(texts))
    start = perf_counter()
    extractor.encode(texts, batch_size=probe_batch_size, normalize=True)
    elapsed = perf_counter() - start
    if elapsed <= 0:
        return None

    rows_per_second = len(texts) / elapsed
    estimated_seconds = total_rows / rows_per_second
    return rows_per_second, estimated_seconds


def enforce_runtime_limit(estimated_seconds: float, max_estimated_seconds: int) -> None:
    if max_estimated_seconds <= 0 or estimated_seconds <= max_estimated_seconds:
        return

    limit = format_seconds(float(max_estimated_seconds))
    estimate = format_seconds(estimated_seconds)
    raise SystemExit(
        "\n".join(
            [
                f"Estimated train embedding time is {estimate}, above the configured limit of {limit}.",
                "Stopping before the expensive full run.",
                "Try one of these:",
                "  uv run python -m semantic_svm.svm_classifier --model-name fast --max-train-rows 20000 --max-dev-rows 5000",
                "  uv run python -m semantic_svm.svm_classifier --model-name fast --max-estimated-seconds 0",
                "  uv run python -m semantic_svm.svm_classifier --model-name large --device cuda",
            ]
        )
    )


def embedding_cache_path(
    output_dir: Path,
    split_name: str,
    csv_path: Path,
    model_name: str,
    text_column: str,
    max_rows: int | None,
) -> Path:
    stat = csv_path.stat()
    fingerprint = "|".join(
        [
            str(csv_path.resolve()),
            str(stat.st_size),
            str(int(stat.st_mtime)),
            model_name,
            text_column,
            str(max_rows or "all"),
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    return output_dir / "embeddings" / f"{split_name}_{digest}.npy"


def build_svm(backend: str, calibration_cv: int, random_state: int, n_jobs: int):
    if backend == "svc":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, random_state=random_state),
        )

    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            LinearSVC(dual=False, max_iter=5000, random_state=random_state),
            method="sigmoid",
            cv=calibration_cv,
            n_jobs=n_jobs,
        ),
    )


def predict_ai_probability(classifier, embeddings: np.ndarray) -> np.ndarray:
    classes = list(classifier.classes_)
    ai_index = classes.index(AI_CLASS_ID)
    return classifier.predict_proba(embeddings)[:, ai_index]


def evaluate(
    classifier, embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    probabilities = predict_ai_probability(classifier, embeddings)
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the semantic rage SVM."
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Embedding model or preset: fast/minilm, bge-small/bge, large/mixedbread, or any Hugging Face model id.",
    )
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--text-column", default="comment")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunksize", type=int, default=4096)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sentence-transformers device, for example cuda or cpu.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-dev-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument(
        "--svm-backend", choices=["linear-svc", "svc"], default="linear-svc"
    )
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--estimate-rows", type=int, default=16)
    parser.add_argument(
        "--max-estimated-seconds",
        type=int,
        default=7200,
        help="Abort before embedding if the speed probe estimates a longer train run. Use 0 to disable.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Estimate train embedding time and exit.",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Create embeddings for the requested splits and stop before label loading, training, and evaluation.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.no_progress
    requested_model_name = args.model_name
    args.model_name = resolve_model_name(args.model_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("Semantic SVM train/dev run")
    log(f"Train CSV: {args.train_csv}")
    log(f"Dev CSV: {args.dev_csv}")
    if args.test_csv is not None:
        log(f"Test CSV: {args.test_csv}")
    log(f"Artifacts: {args.output_dir}")
    if requested_model_name == args.model_name:
        log(f"Embedding model: {args.model_name}")
    else:
        log(f"Embedding model: {requested_model_name} -> {args.model_name}")
    log(f"Embedding batch size: {args.batch_size}; CSV chunk size: {args.chunksize}")

    with progress_step("Loading embedding model", progress):
        extractor = EmbeddingExtractor(
            model_name=args.model_name,
            device=args.device,
            cache_folder=args.model_cache_dir,
        )
    log(f"Embedding device: {extractor.model.device}")
    if str(extractor.model.device) == "cpu" and args.max_train_rows is None:
        log(
            "CPU detected with full train data; expect a long embedding run. Progress bars will show row counts."
        )
    log(f"Embedding progress will update every batch of up to {args.batch_size} rows.")

    train_cache = embedding_cache_path(
        args.output_dir,
        "train",
        args.train_csv,
        args.model_name,
        args.text_column,
        args.max_train_rows,
    )
    dev_cache = embedding_cache_path(
        args.output_dir,
        "dev",
        args.dev_csv,
        args.model_name,
        args.text_column,
        args.max_dev_rows,
    )
    test_cache = None
    if args.test_csv is not None:
        test_cache = embedding_cache_path(
            args.output_dir,
            "test",
            args.test_csv,
            args.model_name,
            args.text_column,
            args.max_test_rows,
        )

    if not train_cache.exists() or args.force_embed:
        train_rows = 0
        with progress_step("Estimating train embedding runtime", progress):
            train_rows = count_rows(
                args.train_csv,
                args.text_column,
                args.chunksize,
                args.max_train_rows,
                progress,
            )
            log(f"Train rows to embed: {train_rows:,}")
            if args.estimate_rows > 0:
                estimate = estimate_embedding_runtime(
                    extractor,
                    args.train_csv,
                    args.text_column,
                    train_rows,
                    args.batch_size,
                    args.estimate_rows,
                )
                if estimate is not None:
                    rows_per_second, estimated_seconds = estimate
                    log(
                        f"Speed probe: about {rows_per_second:.2f} rows/s from the first {min(args.estimate_rows, train_rows)} rows"
                    )
                    log(
                        f"Estimated train embedding time on {extractor.model.device}: about {format_seconds(estimated_seconds)}"
                    )
                    if not args.estimate_only:
                        enforce_runtime_limit(
                            estimated_seconds, args.max_estimated_seconds
                        )
            else:
                log("Speed probe skipped because --estimate-rows was set to 0")
        if args.estimate_only:
            log("Estimate-only mode enabled; stopping before embedding/training.")
            return
    else:
        log(f"Train embedding cache already exists: {train_cache}")
        if args.estimate_only:
            log("Estimate-only mode enabled; train embeddings are already cached.")
            return

    with progress_step("Preparing train embeddings", progress):
        train_embeddings = extractor.embed_csv(
            args.train_csv,
            train_cache,
            text_column=args.text_column,
            batch_size=args.batch_size,
            chunksize=args.chunksize,
            max_rows=args.max_train_rows,
            overwrite=args.force_embed,
            progress=progress,
        )

    with progress_step("Preparing dev embeddings", progress):
        dev_embeddings = extractor.embed_csv(
            args.dev_csv,
            dev_cache,
            text_column=args.text_column,
            batch_size=args.batch_size,
            chunksize=args.chunksize,
            max_rows=args.max_dev_rows,
            overwrite=args.force_embed,
            progress=progress,
        )
    if args.test_csv is not None and test_cache is not None:
        with progress_step("Preparing test embeddings", progress):
            extractor.embed_csv(
                args.test_csv,
                test_cache,
                text_column=args.text_column,
                batch_size=args.batch_size,
                chunksize=args.chunksize,
                max_rows=args.max_test_rows,
                overwrite=args.force_embed,
                progress=progress,
            )

    if args.embed_only:
        manifest_path = args.output_dir / "embedding_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "embedding_model": args.model_name,
                    "requested_model_name": requested_model_name,
                    "text_column": args.text_column,
                    "train_csv": str(args.train_csv),
                    "dev_csv": str(args.dev_csv),
                    "test_csv": (
                        str(args.test_csv) if args.test_csv is not None else None
                    ),
                    "train_cache": str(train_cache),
                    "dev_cache": str(dev_cache),
                    "test_cache": str(test_cache) if test_cache is not None else None,
                    "max_train_rows": args.max_train_rows,
                    "max_dev_rows": args.max_dev_rows,
                    "max_test_rows": args.max_test_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"Embedding-only mode enabled; saved manifest to {manifest_path}")
        return

    with progress_step("Loading train labels", progress):
        train_labels = load_labels(
            args.train_csv,
            args.label_column,
            args.chunksize,
            args.max_train_rows,
            progress,
        )
    with progress_step("Loading dev labels", progress):
        dev_labels = load_labels(
            args.dev_csv, args.label_column, args.chunksize, args.max_dev_rows, progress
        )

    log(f"Train rows: {len(train_labels):,}; dev rows: {len(dev_labels):,}")
    log(f"SVM backend: {args.svm_backend}")
    classifier = build_svm(
        args.svm_backend, args.calibration_cv, args.random_state, args.n_jobs
    )

    with progress_step("Training SVM", progress):
        classifier.fit(train_embeddings, train_labels)

    with progress_step("Evaluating on dev", progress):
        metrics = evaluate(classifier, dev_embeddings, dev_labels)

    model_path = args.output_dir / "semantic_svm.joblib"
    metrics_path = args.output_dir / "dev_metrics.json"
    metadata_path = args.output_dir / "semantic_svm.metadata.json"

    with progress_step("Saving outputs", progress):
        joblib.dump(classifier, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "embedding_model": args.model_name,
                    "requested_model_name": requested_model_name,
                    "svm_backend": args.svm_backend,
                    "text_column": args.text_column,
                    "label_column": args.label_column,
                    "label_to_id": LABEL_TO_ID,
                    "ai_class_id": AI_CLASS_ID,
                    "train_rows": int(len(train_labels)),
                    "dev_rows": int(len(dev_labels)),
                    "metrics_path": str(metrics_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(metrics, indent=2))
    log(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
