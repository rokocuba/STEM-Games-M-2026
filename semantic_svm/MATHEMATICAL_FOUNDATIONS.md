# Semantic SVM Mathematical Foundations

This note explains exactly what the semantic SVM pipeline in this repository does, why each step exists, and what mathematics sits underneath it.

The implementation described here is the one in:

- `semantic_svm/embedding_extractor.py`
- `semantic_svm/svm_classifier.py`
- `semantic_svm/inference_pipeline.py`

## 1. Problem Statement

We observe a dataset of Reddit comments and binary labels:

$$
\mathcal{D} = \{(t_i, y_i)\}_{i=1}^n
$$

where:

- $t_i$ is the raw text of comment $i$
- $y_i \in \{0,1\}$
- $y_i = 0$ means `HUMAN`
- $y_i = 1$ means `AI`

The goal is not to generate text. The goal is to learn a function that maps a comment to a probability:

$$
P(y=1 \mid t)
$$

In this project that probability is called:

- `semantic_rage_probability`

## 2. End-to-End Pipeline

The pipeline has five conceptual stages:

1. Convert raw text into a dense embedding vector.
2. Normalize the embedding vector.
3. Standardize each embedding dimension using training-set statistics.
4. Train a linear support vector machine on those standardized vectors.
5. Convert the SVM decision score into a calibrated probability for the AI class.

Written compactly:

$$
t_i \xrightarrow{\text{embed}} x_i \xrightarrow{\text{normalize}} \hat{x}_i \xrightarrow{\text{scale}} z_i \xrightarrow{\text{SVM}} f(z_i) \xrightarrow{\text{calibration}} P(y_i=1 \mid z_i)
$$

## 3. Why Embeddings Come First

The SVM does not operate on raw strings. It operates on vectors in $\mathbb{R}^d$.

The embedding model defines a function:

$$
e: \text{text} \to \mathbb{R}^d
$$

so that:

$$
x_i = e(t_i)
$$

where $x_i$ is a fixed-length dense vector representation of comment $i$.

The intuition is geometric:

- comments with similar meaning should land near each other in vector space
- comments with different meaning should land farther apart
- an SVM can then try to separate human and AI comments with a hyperplane

In this repository, the default model is currently:

- `sentence-transformers/all-MiniLM-L6-v2`

That model produces 384-dimensional vectors and is chosen because it is cheap to run on CPU. A stronger small-model alternative worth trying is:

- `BAAI/bge-small-en-v1.5`

The much larger `mixedbread-ai/mxbai-embed-large-v1` is available, but on CPU it is too slow for the full dataset.

## 4. Embedding Normalization

The embedding extractor calls the sentence-transformers model with:

- `normalize_embeddings=True`

That means each vector is L2-normalized:

$$
\hat{x}_i = \frac{x_i}{\|x_i\|_2}
$$

This pushes every embedding onto the unit sphere, so:

$$
\|\hat{x}_i\|_2 = 1
$$

Why do this?

- It removes raw vector magnitude as a confounding factor.
- It makes cosine-style geometric comparisons more meaningful.
- It often stabilizes downstream linear models.

If two unit vectors $u$ and $v$ are normalized, then:

$$
u^\top v = \cos(\theta)
$$

So dot products become directly related to angular similarity.

## 5. StandardScaler: Why We Still Scale After Normalization

Even after L2 normalization, different embedding coordinates can still have different empirical means and variances across the dataset.

For each dimension $j$, the training split defines:

$$
\mu_j = \frac{1}{n} \sum_{i=1}^n \hat{x}_{ij}
$$

and

$$
\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{x}_{ij} - \mu_j)^2}
$$

StandardScaler transforms every coordinate to:

$$
z_{ij} = \frac{\hat{x}_{ij} - \mu_j}{\sigma_j}
$$

This matters because a linear SVM is sensitive to feature scale. If one coordinate varies much more than another, it can dominate the optimization even if it is not more informative.

So the practical order is:

1. semantic normalization inside the embedding model
2. statistical scaling across the training dataset

These two operations are not redundant. They solve different problems.

## 6. Linear SVM: The Core Classifier

### 6.1 Binary labels in SVM notation

The dataset uses labels $0$ and $1$, but the standard SVM formulation is written with labels in $\{-1, +1\}$.

So conceptually we define:

$$
\tilde{y}_i = 2 y_i - 1
$$

which maps:

- `HUMAN` from $0$ to $-1$
- `AI` from $1$ to $+1$

### 6.2 Hyperplane and decision score

The linear SVM learns a weight vector $w \in \mathbb{R}^d$ and bias $b \in \mathbb{R}$.

Its decision function is:

$$
f(z) = w^\top z + b
$$

This defines a hyperplane:

$$
w^\top z + b = 0
$$

Predictions are made by the sign of the decision score:

$$
\hat{y} = \text{sign}(f(z))
$$

Interpretation:

- if $f(z) > 0$, the point is on the AI side of the hyperplane
- if $f(z) < 0$, the point is on the HUMAN side
- the magnitude $|f(z)|$ reflects confidence in the separation, before probability calibration

### 6.3 Margin maximization

The SVM does not just want any separating hyperplane. It wants one with a large margin.

For a linear separator, the geometric margin is inversely proportional to $\|w\|$:

$$
\text{margin} = \frac{2}{\|w\|}
$$

So maximizing margin is equivalent to minimizing $\|w\|$.

Large-margin classifiers are attractive because they tend to generalize better and overfit less.

### 6.4 Soft-margin optimization

Real data is not perfectly separable. That is why the soft-margin SVM introduces slack variables $\xi_i \ge 0$.

The primal optimization problem is:

$$
\min_{w,b,\xi} \; \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^n \xi_i
$$

subject to:

$$
\tilde{y}_i (w^\top z_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
$$

The two terms matter for different reasons:

- $\frac{1}{2}\|w\|_2^2$ encourages a large margin
- $C \sum_i \xi_i$ penalizes margin violations and misclassifications

The hyperparameter $C$ controls the tradeoff:

- large $C$: care more about fitting the training data closely
- small $C$: care more about a smoother, more regularized boundary

### 6.5 Hinge-loss view

The same objective can be written in loss form using hinge loss:

$$
\ell_{\text{hinge}}(\tilde{y}, f(z)) = \max(0, 1 - \tilde{y} f(z))
$$

Then the optimization becomes:

$$
\min_{w,b} \; \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^n \max(0, 1 - \tilde{y}_i (w^\top z_i + b))
$$

This is a useful way to think about the model:

- correctly classified points far from the boundary contribute zero loss
- correctly classified points near the boundary still contribute loss
- misclassified points contribute large loss

## 7. Why a Linear Kernel Is Enough Here

The implementation uses a linear decision boundary in embedding space.

That choice is intentional.

The embedding model has already performed a complex nonlinear transformation of the raw text. By the time the data reaches the SVM, it is already living in a semantic feature space. A linear separator in that transformed space is often enough.

Reasons this is a good fit here:

- embeddings already contain rich nonlinear semantic structure
- linear SVMs are fast to train relative to nonlinear kernel SVMs
- linear SVMs are easier to regularize and explain
- with large datasets, exact kernel SVMs become expensive quickly

So the overall system is:

- nonlinear text encoder
- linear classifier on top of that representation

## 8. Why the Repository Uses `LinearSVC` by Default

The code supports two backends:

1. `SVC(kernel="linear", probability=True)`
2. `LinearSVC` wrapped in `CalibratedClassifierCV`

The default is the second one.

Why?

### 8.1 `SVC(kernel="linear")`

This is the exact scikit-learn kernel SVM interface. It is mathematically clean, but its runtime and memory behavior become painful as the number of examples grows.

That makes it fine for small experiments, but not the best default for tens or hundreds of thousands of comments.

### 8.2 `LinearSVC`

`LinearSVC` solves the linear SVM problem with an algorithm specialized for large linear problems. It is much more practical for this repository's dataset sizes.

Its downside is that it gives a decision function, not calibrated probabilities.

That is why the implementation wraps it with `CalibratedClassifierCV`.

## 9. Probability Calibration

The final output required by the project is not just a class label. It is a scalar probability for the AI class.

The SVM decision function $f(z)$ is not a probability. It is an unbounded score.

To turn it into something like:

$$
P(y=1 \mid z)
$$

the code calibrates the raw score with a sigmoid model. Conceptually this is Platt scaling:

$$
P(y=1 \mid f) \approx \frac{1}{1 + \exp(A f + B)}
$$

where $A$ and $B$ are learned from held-out predictions.

This is why the implementation uses:

- `CalibratedClassifierCV(..., method="sigmoid")`

The calibration step matters because the downstream ensemble wants a comparable scalar signal, not just a sign.

## 10. Why Cross-Validated Calibration Is Used

If we fit the sigmoid on the same scores used to fit the SVM, the probabilities can become too optimistic.

Cross-validated calibration helps avoid that:

1. Split the training data into folds.
2. Fit the base classifier on some folds.
3. Predict scores on the held-out fold.
4. Fit the sigmoid using out-of-fold scores.
5. Aggregate the calibrated model.

This produces more realistic probability estimates.

## 11. Evaluation Metrics and What They Mean

The repository computes:

- accuracy
- precision
- recall
- F1
- ROC AUC
- confusion matrix

Suppose the confusion matrix is:

- true negatives: $TN$
- false positives: $FP$
- false negatives: $FN$
- true positives: $TP$

Then:

### 11.1 Accuracy

$$
\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

This is the overall fraction of correct predictions.

### 11.2 Precision

$$
\text{precision} = \frac{TP}{TP + FP}
$$

Interpretation: among all comments predicted as AI, how many were actually AI?

### 11.3 Recall

$$
\text{recall} = \frac{TP}{TP + FN}
$$

Interpretation: among all actually AI comments, how many did we catch?

### 11.4 F1 score

$$
F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

This is the harmonic mean of precision and recall. It is useful when we care about both.

### 11.5 ROC AUC

ROC AUC measures ranking quality across all thresholds, not just the fixed $0.5$ threshold.

Intuitively, it is the probability that a randomly chosen AI comment receives a higher model score than a randomly chosen HUMAN comment.

Higher is better:

- $0.5$ means random ranking
- $1.0$ means perfect ranking

For a probability-producing pipeline, ROC AUC is often one of the most informative metrics.

## 12. Why the Threshold Is 0.5 in Evaluation

After calibration, the code predicts class labels using:

$$
\hat{y} = \mathbf{1}[P(y=1 \mid z) \ge 0.5]
$$

This is a reasonable default, but it is not sacred.

If your team later decides that false accusations of humans are more costly than missed AI comments, you can raise the threshold above $0.5$.

If catching as many AI comments as possible matters more, you can lower it.

## 13. Why Runtime Is Dominated by Embedding, Not SVM Training

In this repository the slow part is almost always the embedding stage.

Roughly:

- embedding cost is $O(n \cdot \text{encoder cost})$
- SVM training cost for a linear model is much closer to $O(n d \cdot \text{iterations})$

where:

- $n$ is the number of comments
- $d$ is the embedding dimension

For MiniLM, $d = 384$. That is small enough that training the linear SVM is cheap once embeddings exist.

That is exactly what you saw in practice:

- embedding took minutes
- SVM fitting took seconds

This is why the code has:

- cached embedding files
- runtime estimation from a small probe sample
- chunked CSV reading
- progress bars

## 14. Why Subset Experiments Make Sense

Because embedding dominates runtime, subset experiments are the right workflow.

Use subsets to compare:

- embedding models
- SVM backend choices
- calibration choices
- thresholds

Once you know which configuration is best, then do the larger run.

This is not cutting corners. It is normal experimental design under compute constraints.

## 15. Mapping the Math to the Code

### `semantic_svm/embedding_extractor.py`

This file implements:

- model loading
- embedding extraction
- normalization through sentence-transformers
- chunked CSV reading
- cached `.npy` embedding files

Mathematically, this is the stage that creates $x_i$ and then $\hat{x}_i$.

### `semantic_svm/svm_classifier.py`

This file implements:

- label loading
- runtime estimation
- StandardScaler
- linear SVM training
- calibration
- dev evaluation
- model and metadata saving

Mathematically, this is the stage that creates $z_i$, trains $f(z)$, and evaluates the resulting probabilities.

### `semantic_svm/inference_pipeline.py`

This file loads the trained classifier and embedding model, then produces:

$$
\text{semantic\_rage\_probability} = P(y=1 \mid t)
$$

for a single raw input string.

## 16. Practical Model Advice for This Repo

If your priority is speed on CPU:

- keep `all-MiniLM-L6-v2`

If your priority is a likely quality improvement while staying small:

- try `--model-name bge-small`

If your priority is maximum semantic quality and you have GPU access:

- try larger models later

The important point is that the SVM math does not change. Only the embedding function $e(t)$ changes.

## 17. Summary in One Equation Chain

The entire pipeline can be summarized as:

$$
t \xrightarrow{e(\cdot)} x \xrightarrow{\text{L2 norm}} \hat{x} \xrightarrow{\text{StandardScaler}} z \xrightarrow{f(z)=w^\top z+b} s \xrightarrow{\text{sigmoid calibration}} P(y=1 \mid t)
$$

where:

- $e(\cdot)$ is the embedding model
- $z$ is the standardized feature vector
- $f(z)$ is the SVM decision score
- the sigmoid calibration converts that score into the final AI probability

That is the full mathematical story behind the semantic SVM used in this project.
