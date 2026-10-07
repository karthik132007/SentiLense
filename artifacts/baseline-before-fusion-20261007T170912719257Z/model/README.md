# Custom SVM + Cardiff RoBERTa fusion

`model_metadata.json` describes the deployed `svm_roberta_fusion` model.
`fusion_head.joblib` stores a learned StandardScaler/multinomial logistic
stacking layer and its disjoint sigmoid calibrator. It combines the SVM's
log odds with RoBERTa's three log probabilities. Both expert outputs affect
the final scores; this is trained stacking, not a fixed average or a keyword rule.

The frozen TF-IDF LinearSVC expert is in `svm/`. Its own metadata/checksum and
original binary evaluation remain available. The requested frozen
[cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
checkpoint is in `roberta/`, pinned to revision
`daefdd1f6ae931839bce4d0f3db0a1a4265cd50f`. Weights are saved as safetensors.
The separately trained IMDb review SVM is in `reviews/`.

The trained head has classes 0=negative, 1=neutral, 2=positive. Inference returns
all three probabilities and selects the largest; confidence is that selected
score. Neutral is learned. The binary review model still uses abstention.
Blank inputs are rejected. Model scores are estimates, not guaranteed accuracy.

## Training and evaluation

```bash
uv sync --frozen --extra api
uv run --frozen --extra api python scripts/train_fusion.py --svm-dir model/svm --download --output-dir /tmp/sentilense-fusion-retrain
uv run --frozen --extra api python scripts/install_fusion.py --trained-root /tmp/sentilense-fusion-retrain
PYTHONPATH=backend uv run --frozen --extra api pytest backend/tests tests -q
```

Download the pinned TweetEval files with `scripts/train_tweets.py --download
--output-dir /tmp/svm-retrain` if needed. This also rebuilds the binary expert;
pass `/tmp/svm-retrain/model` to fusion training to use it.

Retain official split assignments, remove normalized conflicting-label groups
and duplicates, with held-out rows taking precedence over training duplicates.
SVM fitting uses official train only. The upstream RoBERTa checkpoint is already
TweetEval-finetuned and stays frozen. Split official validation 50/25/25 with
seed 42: 1,000 head-training, 500 calibration and 500 selection records. Select
C from 0.1/1/10 using macro F1, then log loss as tie-breaker. Do not refit.
Evaluate the frozen fusion and RoBERTa alone on the same cleaned official
three-class test set. Upstream model selection may have used official validation;
this is not a new unseen dataset for RoBERTa. Regression sentences are evaluation
only, and are absent from the fitting/calibration/selection splits.

Evidence is in `../artifacts/fusion/`: complete evaluation, split IDs, held-out
probabilities and expert-feature caches. `../artifacts/sentiment_regressions.json`
records checks for complaints, negation, positive profanity and factual neutral
text. Historical models/evidence are preserved under `../artifacts/baseline-*/`.

## Deployment

Deploy the complete `model/` directory, the `sentiment_model` package and locked
dependencies. Checksums cover every expert and head file; corrupt or missing
components cause startup failure. No alternate prediction silently substitutes
for RoBERTa. `SENTILENSE_MODEL_DIR` selects another complete deployment.

The tokenizer/model load with `local_files_only=True` and `trust_remote_code=False`.
Inference uses CPU float32, four threads and a lock around RoBERTa batches.
Username/link placeholders follow Cardiff's model instructions and preserve
case. For long text, average RoBERTa scores over overlapping 512-token windows
(64-token overlap); the SVM reads full text. Segment/source summaries average
all class scores. These document aggregations do not establish webpage/topic
stance accuracy. Sarcasm, domain shift and non-English content remain limitations.
