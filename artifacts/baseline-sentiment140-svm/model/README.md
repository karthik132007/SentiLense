# SentiLense linear SVM

The active text/general-page model is **word/character TF-IDF + LinearSVC**,
with a sigmoid calibrator fitted on a separate calibration partition.
The complete preprocessing, TF-IDF, frozen SVM, and calibrator are saved in
`sentiment_pipeline.joblib`. Its SHA-256 is verified against `model_metadata.json`
at load. Inference uses `sentiment_model.predict_sentiment`.

## Reproduce

```bash
uv sync --frozen --extra api
uv run python scripts/train.py --data data/archive.zip
PYTHONPATH=backend uv run --extra api pytest backend/tests tests -q
```

Use `--output-dir /tmp/sentilense-svm` to train without replacing deployed files.
`--max-records 100000` makes a stratified smoke subset; the saved production
model uses all 1,541,299 cleaned records, not a smoke subset.
Python and dependency versions are pinned in `pyproject.toml` and `uv.lock`.

## Data and evaluation

Sentiment140 has 1,600,000 headerless Latin-1 rows, labeled
0 (negative) or 4 (positive). Normalized duplicate texts are collapsed;
every group with conflicting labels is excluded before splitting. Dataset
inspection and archive checksum are saved in `../artifacts/dataset_inspection.json`.

Seed 42; stratified 70/5/5/20 train/calibration/validation/test:

| Partition | Records | Negative | Positive |
| --- | ---: | ---: | ---: |
| Train | 1,078,909 | 541,992 | 536,917 |
| Calibration | 77,065 | 38,713 | 38,352 |
| Validation | 77,065 | 38,714 | 38,351 |
| Test | 308,260 | 154,855 | 153,405 |

Vocabulary, IDF and SVM are fitted only on train. Sigmoid probability calibration
uses only calibration. C is selected from 0.5/1/2 by validation accuracy, with
Brier score breaking ties. Uncertainty thresholds are selected on validation
for >=90% empirical precision per accepted class. Test is evaluated after these
choices, with no refit. Calibration and test are not used for fitting the SVM.

The selected C is 0.5; fitted vocabularies contain
300,000 word unigrams/bigrams and
50,000 within-word character 4/5-grams.
Features use min_df=3 and sublinear TF. LinearSVC uses squared hinge loss,
L2 regularization, dual='auto', tol=1e-4, max_iter=3000. It converged without
warnings. Character features support spelling variation; inference contains
no profanity lists or hardcoded polarity rules.

Shared normalization repairs Unicode, unescapes HTML, lowercases, replaces
URL/mention identity, and retains negation, contractions, emoticons, emojis,
hashtags and punctuation. No stopwords are removed.

| Binary held-out metric | Result |
| --- | ---: |
| Accuracy | 0.824288 |
| Positive precision | 0.817227 |
| Positive recall | 0.833278 |
| Positive F1 | 0.825174 |
| ROC-AUC | 0.902969 |
| Brier score | 0.125009 |

## Inference policy

P(positive) <= 0.26 means negative; >=
0.75 means positive; the intermediate band is uncertain
(`neutral` in the API). Neutral is not a learned third class. Confidence is the
larger calibrated binary probability, not a probability of neutrality.
Calibration on tweets does not guarantee calibrated scores on other domains.
Inputs with zero recognized features return 0.5/0.5 and neutral. Blank input
raises ValueError. Reloaded predictions match the fitted pipeline exactly.

| Text | Label | P(positive) |
| --- | --- | ---: |
| I absolutely love this! | positive | 0.967651 |
| I hate this product. | negative | 0.035548 |
| The product arrived yesterday. | neutral | 0.400977 |

```python
from sentiment_model import predict_sentiment
result = predict_sentiment('I absolutely love this!')
```

Deploy the joblib and metadata together with the `sentiment_model` Python
package and locked dependencies; restart the server after replacement.
`SENTILENSE_MODEL_DIR` or `model_dir=` selects another model directory.

Saved evidence in `../artifacts/` includes split source rows (including
calibration), held-out predictions, metrics, threshold search, ROC coordinates,
reload checks and inference examples. The prior logistic model and evidence are
preserved in `../artifacts/baseline-logistic/`; comparison is in
`../artifacts/svm_comparison.json`. The reported failing sentence was checked
after selection and was not added to training or used to choose settings.

The reported negative sentence remains incorrectly classified as positive
(P(positive)=0.753779). Its desired negative classification is tracked as a
strict expected failure in the backend regressions. Tweet test accuracy is
82.43%, versus 82.62% for the prior logistic baseline; this migration is not
presented as an accuracy improvement or a fix for that complaint.

Historical weakly labeled English tweets and a random split do not establish
article, topic-stance, sarcasm, chronological or user-disjoint accuracy.
