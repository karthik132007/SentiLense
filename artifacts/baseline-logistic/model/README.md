# SentiLense sentiment model

This directory contains the trained ML assets. Backend code can import
`predict_sentiment` from `sentiment_model`; this project does not implement
an API, frontend, or scraper.

## Run and reproduce

Python 3.12+ and `uv` are required. Training used Python 3.14.7. Dependency
versions are pinned in `pyproject.toml` and resolved in `uv.lock`.

```bash
uv sync --frozen
uv run python scripts/train.py --data data/archive.zip
uv run python -m unittest discover -s tests -v
```

The script accepts a CSV, ZIP containing one CSV, or directory containing one
CSV/ZIP. Its default data location is `/data` when available, otherwise the
repository's `data/`. In this session `/data` was absent; the existing archive
was found at `data/archive.zip`. The CSV is read directly from the archive.
The full dataset was practical on CPU, so no subset or alternative model was
needed. Fit took 121.5 seconds; the complete run took 190.9 seconds.

To verify a smaller stratified subset without overwriting the final artifacts:

```bash
uv run python scripts/train.py --data data/archive.zip \
  --max-records 100000 --output-dir /tmp/sentilense-smoke
```

The Colab VS Code extension is installed, but this agent session has no callable
VS Code kernel-control interface and no local CUDA GPU. Training was completed
on CPU. Opening a notebook and choosing **Select Kernel > Colab > Auto Connect**
is the extension's documented connection workflow and may require sign-in:
[official Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab).

## Verified input and cleaning

- Filename inside the archive: `training.1600000.processed.noemoticon.csv`.
- 1,600,000 rows; no header. Every row has six fields. Semantic column names
  are `target, id, date, flag, user, text`; they are inferred from the observed
  values and positions, rather than read from a nonexistent header.
- Latin-1 (`iso-8859-1`) decoding, with strict UTF-8 validation failing near
  byte 35,888,930. CP1252 decoding is also valid; single-byte encodings cannot
  always be distinguished uniquely from bytes. `ftfy` repairs mojibake and
  C1 characters after decoding.
- Labels are exactly `0` (negative) and `4` (positive), 800,000 each.
- No empty/whitespace-only fields in any column. All IDs are numeric and all
  dates match the observed Sentiment140 date format. Every flag is `NO_QUERY`.
- 0 exact duplicate rows; 18,534 repeated raw text rows; 55,374 repeated
  normalized text rows, including 3,327 normalized groups with conflicting
  labels. Duplicate counts overlap and must not be added together.
- Deduplication keeps the first normalized text with consistent labels and
  excludes every group with conflicting labels **before splitting**.
- 58,701 records removed; 1,541,299 retained: 774,274 negative and 767,025 positive.

`../artifacts/dataset_inspection.json` contains the full inspection and the
archive SHA-256 for provenance. `split_indices.npz` stores zero-based row
positions in the original CSV, so the split is auditable.

## Pipeline and evaluation protocol

The saved scikit-learn `Pipeline` includes all training-time text normalization,
word unigram/bigram TF-IDF, and Logistic Regression. Preprocessing lowercases,
repairs malformed Unicode with `ftfy`, unescapes HTML, normalizes Unicode to NFC,
replaces URLs and mentions with placeholders, and cleans whitespace/controls.
It retains negation, emoticons, emoji, hashtags, and punctuation. No stopwords
are removed and no stemming is applied. The custom tokenizer preserves
contractions and emoticons as tokens.

TF-IDF uses `min_df=3`, sublinear term frequency, and at most 300,000 features.
Logistic Regression uses `solver="liblinear"`, `C=4`, `max_iter=200`, and
`tol=1e-4`. The model converged in seven iterations without a convergence warning.
IDF, vocabulary, and classifier coefficients are learned from training only.

The random seed is 42 and both splits are stratified:

| Partition | Fraction | Records | Negative | Positive |
| --- | ---: | ---: | ---: | ---: |
| Train | 70% | 1,078,909 | 541,992 | 536,917 |
| Validation | 10% | 154,130 | 77,427 | 76,703 |
| Test | 20% | 308,260 | 154,855 | 153,405 |

The model is not refitted on validation/test records after evaluation.
`records_used` in metadata includes all three partitions;
`records_trained` is the actual classifier training count.

Binary test metrics use a decision threshold of 0.5. Precision, recall, and F1
below refer to the positive class; the complete per-class report is saved too.

| Metric | Test result |
| --- | ---: |
| Accuracy | 0.826163 |
| Precision | 0.819991 |
| Recall | 0.833702 |
| F1 | 0.826789 |
| Macro F1 | 0.826161 |
| ROC-AUC | 0.905004 |
| Brier score | 0.123733 |

Confusion matrix (rows actual, columns predicted):

| | Negative | Positive |
| --- | ---: | ---: |
| Negative | 126,779 | 28,076 |
| Positive | 25,511 | 127,894 |

## Neutral as uncertainty

No neutral training class was created. The inference policy is:

```text
P(positive) <= 0.26          -> negative
0.26 < P(positive) < 0.75   -> neutral (uncertain / abstain)
P(positive) >= 0.75         -> positive
```

Thresholds were selected **only on validation data**. A grid in increments of
0.01 maximizes accepted predictions for each class subject to at least 90%
empirical precision and at least 5% of validation records accepted per class
(with a minimum of 100). The entire search is saved in
`../artifacts/neutral_threshold_selection.json`.

On validation, negative/positive precision is 90.04%/90.15%. On test,
negative/positive precision is 90.15%/90.29%. Test coverage is 72.23%
(222,666 accepted predictions), with 90.22% accepted accuracy; 85,594 records
(27.77%) fall into the uncertainty band.

Those are binary selective-classification measurements. This dataset cannot
establish true neutral precision, recall, or F1. Logistic probabilities are not
independently calibrated. Returned `confidence` means the larger **binary**
score, including for neutral outputs; it is not a probability of true neutrality.
Inputs with no recognized TF-IDF features return neutral with both scores 0.5.
Empty/whitespace-only text raises `ValueError`; non-string input raises `TypeError`.

## Backend import

Run from the repository or install this Python project with the locked dependencies:

```python
from sentiment_model import predict_sentiment

result = predict_sentiment("I absolutely love this!")
# {
#   "sentiment": "positive",
#   "confidence": 0.9863024546716835,
#   "scores": {
#     "positive": 0.9863024546716835,
#     "negative": 0.013697545328316463
#   }
# }
```

Default model paths are resolved relative to the Python package's repository
location, independently of the current working directory. For a relocated or
installed package, set `SENTILENSE_MODEL_DIR` to an absolute model-directory
path or pass `model_dir=` to `predict_sentiment`.

Deploy these files together:

- `model/sentiment_pipeline.joblib`
- `model/model_metadata.json`
- The `sentiment_model/` Python package, including `preprocessing.py`.
- Dependencies in `pyproject.toml` / `uv.lock`.

The joblib pipeline references the importable preprocessing functions. Loading
is lazy and cached once per directory; the model SHA-256 is verified against
metadata before first loading. Restart the backend after replacing artifacts.
Only load the trusted locally generated joblib file.

Actual saved-model example outputs:

| Text | Sentiment | P(positive) | Binary confidence |
| --- | --- | ---: | ---: |
| I absolutely love this! | positive | 0.986302 | 0.986302 |
| I hate this product. | negative | 0.017571 | 0.982429 |
| The product arrived yesterday. | neutral | 0.411290 | 0.588710 |

## Saved evidence

`../artifacts/` contains dataset inspection, evaluation JSON, classification
report, confusion-matrix CSV, validation threshold search, split row positions,
held-out probabilities/labels, complete ROC coordinates, and inference examples.
The training script verifies that a reloaded joblib model reproduces predictions
on 100 held-out records exactly. Tests independently recompute evaluation
metrics, check split disjointness, and exercise shared preprocessing/inference.

Historical tweet labels and a random tweet split do not measure future,
user-disjoint, or product-review generalization. Sarcasm and domain shift can
still produce confident errors. Neutral here is the documented uncertainty
policy, not evidence that a text is semantically neutral.
