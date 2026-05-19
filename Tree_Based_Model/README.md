## Punctuation analysis

Run the analysis script inside `Tree_Based_Model/` to compare HUMAN vs AI comments in `data/train.csv`:

```bash
python Tree_Based_Model/main.py
```

The script now writes three plots and three CSV summaries:

- `Tree_Based_Model/artifacts/punctuation_hyper_correctness_by_label.png`
- `Tree_Based_Model/artifacts/punctuation_hyper_correctness_summary.csv`
- `Tree_Based_Model/artifacts/macro_structural_rules_by_label.png`
- `Tree_Based_Model/artifacts/macro_structural_rules_summary.csv`
- `Tree_Based_Model/artifacts/expected_characters_after_syntagms_by_label.png`
- `Tree_Based_Model/artifacts/expected_characters_after_syntagms_summary.csv`

It measures em dashes, semicolons, Oxford-comma-style lists, and apostrophe-bearing contractions.
It also measures balanced counter-weight phrases, rule-of-three structure, bullet/numbered list defaults, and wrap-up phrases.
The third chart measures expected punctuation after specific words and syntagms such as interjections, transition phrases, polite refusals, greetings, and rhetorical pivots.

## Sentence fact-check

Run the dedicated fact-check script to build a 20-panel overview, a CSV verdict table, and one plot per category:

```bash
python Tree_Based_Model/sentence_category_factcheck.py
```

The script writes:

- `Tree_Based_Model/artifacts/sentence_category_factcheck_overview.png`
- `Tree_Based_Model/artifacts/sentence_category_factcheck_summary.csv`
- `Tree_Based_Model/artifacts/sentence_category_factcheck_individuals/`

The CSV uses a plain-language `label` column and then `human_frequency` and `ai_frequency` next to it.

## Correlation analysis

Run the correlation script to see which of the 20 sentence features move together and which ones track the AI label:

```bash
python Tree_Based_Model/sentence_category_correlation.py
```

It writes a heatmap, a ranked pairwise CSV, and a label-correlation CSV in `Tree_Based_Model/artifacts/`.

## Pair analysis

Run the pair analysis script when you want to inspect two features directly, for example sentences per paragraph versus question mark present:

```bash
python Tree_Based_Model/sentence_pair_analysis.py
```

It writes a labeled scatter plot and a compact CSV summary in `Tree_Based_Model/artifacts/`.

## Tree model

Train the default boosted tree model on `data/train.csv` and evaluate it on `data/dev.csv`:

```bash
python Tree_Based_Model/tree-based.py
```

It uses histogram gradient boosting, prints accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and a confusion matrix, and saves a permutation-importance plot in `Tree_Based_Model/artifacts/`.

For an exact root-to-leaf walk on a small number of cases, use the single decision-tree mode:

```bash
python Tree_Based_Model/tree-based.py --model decision-tree --explain-count 2
```

That mode keeps memory use lower and prints the full node-by-node path for the first two cases from `data/dev.csv` by default.

To generate a presentation-ready image for one example, add `--path-image-output`:

```bash
python Tree_Based_Model/tree-based.py --model decision-tree --path-image-output Tree_Based_Model/artifacts/tree_walk_example.png --path-case-index 0
```

That writes both `Tree_Based_Model/artifacts/tree_walk_example.png` and a matching `Tree_Based_Model/artifacts/tree_walk_example.svg` so you can paste the figure into slides or edit it as vector art.
