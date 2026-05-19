# 🏗️ REVISED SYSTEM PROMPT & ARCHITECTURE BLUEPRINT

**Component:** Semantic Outrage Detector (Sub-Model 3 of 3)
**Target:** Detecting "Manufactured Outrage" and Engagement-Bait AI Bots on Reddit
**Architecture Role:** This pipeline acts purely as the "Deep Semantic Expert." It will process raw text, extract deep semantic vectors, and output a single condensed probability signal to the overarching ensemble Meta-Classifier.

## 1. Project Context & Objectives

We are building a machine learning ensemble to detect AI-generated Reddit bots using _only_ comment text. The target is **Engagement-Inducing Bots** (rage-bait).
Your task is to build **Sub-Model 3: The Semantic Pipeline.** You will not build the final Meta-Classifier, nor will you calculate surface-level text statistics (readability, punctuation counts, etc.), as those are handled by another part of the system.

The core hypothesis is that _Genuine Human Anger_ occupies a fundamentally different geometric subspace than _Calculated AI Rage-Bait_ in a high-dimensional language model embedding space.

## 2. Technical Stack & Required Libraries

- **Language:** Python 3.10+
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` by default for local CPU runs. `mixedbread-ai/mxbai-embed-large-v1` remains an optional large model for GPU runs or smaller experiments.
- **Classification:** `scikit-learn` (specifically `SVC` with `probability=True` and `StandardScaler`).
- **Data Handling:** `pandas`, `numpy`.

## 3. Pipeline Architecture & Implementation Steps

The AI agent must implement the following 3-step pipeline:

### Step 1: Deep Semantic Embedding

**Objective:** Capture the deep contextual and geometric meaning of the text.

- Load the `sentence-transformers/all-MiniLM-L6-v2` model by default, or `mixedbread-ai/mxbai-embed-large-v1` when compute allows it.
- Pass the raw Reddit comments through the model to generate static, frozen embedding vectors.
- _Constraint:_ Do not build a neural network to fine-tune these embeddings. Treat them as frozen feature extractors.

### Step 2: Support Vector Machine (SVM) Classification

**Objective:** Find the geometric hyperplane that separates human anger from AI rage-bait in the semantic embedding space.

- Build a scikit-learn pipeline: `make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))`.
- Train the SVM on the training split of the 1024-d embeddings and their binary labels (0=Human, 1=AI).
- _Why SVM?_ SVMs are mathematically optimal for finding hyperplanes in high-dimensional, sparse spaces like embeddings without overfitting.

### Step 3: Output Synthesis (Handoff to Meta-Classifier)

**Objective:** Package the finding into a single, high-value scalar for the final ensemble Meta-Classifier.
For any new comment processed by the system, the final output of this pipeline must be exactly one value:

- `semantic_rage_probability` (Float 0.0 - 1.0): The output of the SVM's `predict_proba()` function for the "AI" class.

## 4. Deliverables Required from the Agent

Please generate the following:

1. **`semantic_svm/embedding_extractor.py`**: The script containing the code to load the embedding model and embed the text into fixed-size vectors.
2. **`semantic_svm/svm_classifier.py`**: The script to train the SVM on the embeddings, calibrate the probabilities, and save the model using `joblib`.
3. **`semantic_svm/inference_pipeline.py`**: A wrapper class `SemanticRageAnalyzer` with a `predict_probability(text)` method. This method must take a raw string, run it through the embedding model, pass the embedding to the saved SVM, and return the single float `semantic_rage_probability`.
