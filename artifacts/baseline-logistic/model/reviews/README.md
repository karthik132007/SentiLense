# Movie-review document model

An additional academic TF-IDF + Logistic Regression model for movie-review
pages. The original Sentiment140 pipeline remains the text/general-page
baseline. All labels come from learned model probabilities; URLs and negative
phrases are not hardcoded to a verdict.

Data source: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/),
[Maas et al., ACL 2011](https://aclanthology.org/P11-1015/).

```bash
uv run python scripts/train_reviews.py --data data/aclImdb_v1.tar.gz
```

The official archive SHA-256 is stored in `model_metadata.json`. The trainer
streams its UTF-8 labeled reviews without extracting tar files. It drops 98
duplicate normalized training reviews, 201 test duplicates, and 123 test rows
matching training content. Deduplication keeps the first occurrence. Only the
original labeled partitions are used; the unlabeled partition is excluded.

The 24,902 retained official training reviews are split stratified, seed 42,
into 19,921 training and 4,981 validation reviews. The held-out official test
has 24,676 reviews. Validation is not movie-disjoint. Features fit only on train;
`C` is selected from 1/4/12 by validation accuracy (Brier score breaks ties).
The selected C is 12; no refitting occurs after selection. The reported failing
article was not added to training, validation, or threshold selection.

The saved pipeline includes shared Unicode/HTML entity/URL/mention/whitespace
normalization and the sentiment-preserving tokenizer. HTML `br` elements in
raw corpus reviews are replaced with whitespace during ingestion. Backend
inputs are already extracted plain text. Word unigrams/bigrams use `min_df=2`,
200,000 maximum features and sublinear TF. Logistic Regression uses liblinear,
`max_iter=300`, and seed 42. Reloaded probabilities match the original exactly.

| Binary official-test metric | Result |
| --- | ---: |
| Accuracy | 0.906022 |
| Positive-class F1 | 0.906646 |
| ROC-AUC | 0.966424 |

The full precision/recall/report/confusion matrix and validation candidates are
in `artifacts/reviews/evaluation.json`. Split filenames, held-out probabilities,
and the complete threshold search are saved alongside it. Tests independently
recompute metrics and check split separation.

Neutral remains uncertainty, not a trained class. Thresholds selected on
validation only, targeting >=90% empirical precision for each accepted class:
negative <=0.49, positive >=0.51, otherwise uncertain/neutral. IMDb reviews are
polarized, so this narrow band does not establish true neutral detection.
Confidence is the larger binary probability and is not independently calibrated.

Backend routing requires a review headline plus movie/film wording, or at least
three cinema vocabulary terms in its body. Routing is disclosed in the API and
UI. Overall review sentiment runs once on the complete extracted document
including its title; segment labels are exploratory and do not set the verdict.
Benchmark review accuracy does not establish accuracy on web articles, sarcasm,
other domains, or languages. Non-review pages retain the tweet model.

```python
from sentiment_model.reviews import predict_review
prediction = predict_review('The film is a tedious mess with terrible acting.')
```

Deploy this directory's joblib and metadata together with the `sentiment_model`
package. Checksums are verified at load. `SENTILENSE_REVIEW_MODEL_DIR` selects a
custom absolute directory. Restart the server after replacing either artifact.
