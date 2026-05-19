# Semantic SVM

This directory contains the semantic SVM component for detecting AI-generated Reddit comments from text only.

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. It is local, free, much faster on CPU than Mixedbread, and produces compact 384-dimensional embeddings. If you want a likely quality upgrade while staying in the small-model class, try `--model-name bge-small` for `BAAI/bge-small-en-v1.5`. The larger Mixedbread model is still available with `--model-name large`, but it is not practical for the full train split on a CPU-only laptop.

Train on `data/train.csv` and evaluate on `data/dev.csv`:

```bash
uv run python -m semantic_svm.svm_classifier
```

Create fresh train/dev/test embeddings without training the SVM:

```bash
uv run python -m semantic_svm.svm_classifier --embed-only --test-csv data/test.csv --max-estimated-seconds 0 --force-embed
```

Useful shorter runs while developing:

```bash
uv run python -m semantic_svm.svm_classifier --max-train-rows 100 --max-dev-rows 50 --calibration-cv 2
uv run python -m semantic_svm.svm_classifier --model-name bge-small --max-train-rows 20000 --max-dev-rows 5000
uv run python -m semantic_svm.svm_classifier --estimate-only
uv run python -m semantic_svm.svm_classifier --device cuda --batch-size 128
uv run python -m semantic_svm.svm_classifier --model-name large --device cuda
uv run python -m semantic_svm.svm_classifier --svm-backend svc --max-train-rows 5000
```

The script prints timed steps and progress bars for model loading, row counting, embedding, label loading, training, evaluation, and saving. Before full embedding, it runs a small speed probe and stops if the estimated training embedding time is above `--max-estimated-seconds` (default: 7200 seconds). Use `--max-estimated-seconds 0` only when you intentionally want to allow a long run.

In `--embed-only` mode, the script writes train/dev/test embedding caches plus `semantic_svm/artifacts/embedding_manifest.json` and stops before SVM training.

External embedding APIs are not recommended for this project unless you specifically need them. They add cost, rate limits, reproducibility issues, and data-sharing concerns. A smaller local model is the safer default.

Embeddings, metrics, and the trained model are saved under `semantic_svm/artifacts/` by default.

For inference, use `SemanticRageAnalyzer.predict_probability(text)` from `semantic_svm/inference_pipeline.py`. It returns the AI-class `semantic_rage_probability` as one float.

For the detailed math behind the pipeline, see `semantic_svm/MATHEMATICAL_FOUNDATIONS.md`.
