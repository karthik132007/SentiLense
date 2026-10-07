# Movie-review linear SVM

The saved review pipeline combines shared preprocessing, word/character TF-IDF,
LinearSVC, and sigmoid probability calibration. It is used on the complete
extracted movie-review document plus title. Segment scores are exploratory.
Non-review text uses the separate Sentiment140 SVM.

Data: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
The local official archive checksum is stored in `model_metadata.json`.
Unlabeled records are excluded. Normalized duplicates and test matches to
training are removed before splitting.

```bash
uv run python scripts/train_reviews.py --data data/aclImdb_v1.tar.gz
```

`--output-dir /tmp/sentilense-svm` trains into an isolated output directory.
The official training partition is stratified, seed 42: 19,921
classifier-fit records, 2,490 disjoint calibration
records, and 2,491 validation records. There are
24,676 official test reviews. C is selected from 0.5/1/2 on
validation only; selected C=0.5. No refit uses held-out data.
TF-IDF uses word unigrams/bigrams (max 200,000), character 4/5-grams (max 50,000),
min_df=2 and sublinear TF. Sigmoid calibration uses the frozen fitted SVM.

| Official-test binary metric | Result |
| --- | ---: |
| Accuracy | 0.899781 |
| Positive F1 | 0.899296 |
| ROC-AUC | 0.963333 |

Validation selects uncertainty thresholds: negative <=
0.49, positive >= 0.51,
otherwise uncertain (`neutral`). Neutral is not a trained class. Confidence
is the larger calibrated binary probability, not guaranteed correctness.

Metrics, split filenames (including calibration), held-out predictions,
validation candidates and threshold search are saved in `artifacts/reviews/`.
Reloaded probabilities match exactly. The previous logistic artifact and
evidence remain under `artifacts/baseline-logistic/`.

Routing requires a review headline with movie/film wording, or a review headline
and at least three cinema terms in the body. This selects the model, not a label.
Movie-review benchmark results do not establish article or sarcasm accuracy;
the reported article was not added to training or used for model selection.

Deploy joblib and metadata together with the Python package. The checksum is
verified at load. `SENTILENSE_REVIEW_MODEL_DIR` relocates this directory.
