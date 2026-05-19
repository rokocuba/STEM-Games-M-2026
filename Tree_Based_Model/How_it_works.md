# How It Works

This project trains a tree-based classifier to predict whether a comment is HUMAN or AI using sentence-level and paragraph-level text signals.

## 1. Input Data

The model reads `data/train.csv` for training and `data/dev.csv` for evaluation.
Each row has:

- `comment`: the text to analyze
- `label`: `HUMAN` or `AI`

The model does not use raw word embeddings or a language model. It converts each comment into a small set of numeric features that describe punctuation and structure.

## 2. Features

The features come from `sentence_category_factcheck.py`. For each comment, the script computes 20 heuristic measurements such as:

- sentences per paragraph
- words per paragraph
- question mark present
- apostrophe present
- semicolon or colon present
- sentence length standard deviation
- consecutive sentence gap
- long or short sentence presence
- specific word or syntagm signals like `although`, `however`, `because`, `this`, `et`, and `others/researchers`

Each comment becomes a 20-value numeric vector.

## 3. Training

`tree-based.py` supports two model modes:

- `boosted`: the default, using `HistGradientBoostingClassifier`
- `decision-tree`: a single `DecisionTreeClassifier` for exact tree-path explanations

The boosted model is the default because it gives better accuracy on `data/dev.csv`.
The single decision tree is available when you want a literal root-to-leaf walk for one or two examples.

## 4. Prediction

After training, the model evaluates each dev example and produces:

- predicted label
- probability for the AI class
- accuracy, balanced accuracy, precision, recall, F1, ROC AUC
- confusion matrix

The default boosted model is better for classification performance.
The single decision tree is better for explanation because every prediction follows one explicit path of threshold tests.

## 5. How the Tree Makes a Decision

For the single-tree mode, each node checks one feature against a threshold.

Example:

```text
Sentences per paragraph <= 2.595 ?
```

If the value is below or equal to the threshold, the tree goes left. Otherwise it goes right.
That repeats until the tree reaches a leaf.

At the leaf, the model uses the class counts stored in that leaf to decide whether the example is more likely HUMAN or AI.

## 6. Explain Mode

You can print the full decision walk for the first two examples with:

```bash
python Tree_Based_Model/tree-based.py --model decision-tree --explain-count 2
```

You can also generate a presentation-ready image for a single example with:

```bash
python Tree_Based_Model/tree-based.py --model decision-tree --path-image-output Tree_Based_Model/artifacts/tree_walk_example.png --path-case-index 0
```

That creates both a PNG and an SVG in `Tree_Based_Model/artifacts/`.

## 7. Why There Are Two Model Types

The project uses two tree styles for different goals:

- the boosted model for stronger predictive performance
- the single decision tree for exact explanation and slide-friendly visualization

If you want to understand why one specific comment was classified a certain way, use the single decision tree.
If you want the best general dev score, use the boosted model.

## 8. Output Folder

All generated plots and CSV summaries now go into `Tree_Based_Model/artifacts/`.
The `data/` root is kept for the split files only:

- `data/train.csv`
- `data/dev.csv`
- `data/test.csv`
