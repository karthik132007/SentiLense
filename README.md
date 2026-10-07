# SentiLense

An academic sentiment analysis application combining an Angular frontend,
FastAPI keyword/text/webpage analysis, a custom SVM + RoBERTa fusion, and an IMDb review SVM. Enter a keyword such as **antigravity**: the app
searches the web, scrapes readable public pages, and shows a sentiment summary,
individual source results, links, and exportable JSON. Predictions use saved trained models. No mock predictions are used.

## Run the complete application

From the repository root, using Python 3.12+, `uv`, and Node.js supported by
the Angular version in `frontend/package.json`:

```bash
uv sync --frozen --extra api
cd frontend
npm ci
npm run build
cd ..
uv run --frozen --extra api python scripts/serve.py
```

Open **http://127.0.0.1:8000**. This one server serves the production frontend
and API, so frontend requests to `/api` use the same origin. API documentation
is at http://127.0.0.1:8000/docs and readiness at `/api/health` must report
`model_loaded: true` and `model_backend: sentiment_model`.

The trained model is already present. Model artifacts are validated at server
startup; a missing or corrupt model causes a startup error rather than fabricated
predictions. To rebuild the model, run:

```bash
uv run --frozen --extra api python scripts/train_fusion.py --svm-dir model/svm --download --output-dir /tmp/sentilense-fusion-retrain
uv run --frozen --extra api python scripts/install_fusion.py --trained-root /tmp/sentilense-fusion-retrain
```

For frontend development, start the backend command with `--reload` and run
`npm start` in `frontend/`. The Angular development configuration targets
http://127.0.0.1:8000; its production configuration uses same-origin `/api`.

## Custom SVM + RoBERTa model

Text, general webpages and keyword passages use a learned three-class fusion:

1. A frozen word/character TF-IDF LinearSVC trained on TweetEval polarity labels.
2. The exact frozen [Cardiff Twitter RoBERTa checkpoint](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).
3. A custom stacking head trained on their scores, with separate calibration
   and model selection data. It produces negative, neutral and positive scores.

Neutral is now a trained class for this fusion. Confidence is the selected
class's score. The interface shows all three probabilities and identifies the
fusion. The separate IMDb movie-review SVM retains its binary uncertainty policy.

Model files are in `model/`: `svm/`, `roberta/`, `fusion_head.joblib` and
`model_metadata.json`. All component checksums are checked at startup. RoBERTa
loads locally from safetensors; running predictions requires no Hugging Face
service or network connection. CPU inference uses four PyTorch threads and
batches page segments. Long text uses overlapping 512-token RoBERTa windows,
while the SVM reads the complete text.

The stacking head uses 1,000 official TweetEval validation examples for fitting,
500 for probability calibration and 500 for selecting regularization. The SVM
uses official train data; RoBERTa retains its upstream TweetEval finetuning.
Evaluation uses the cleaned official three-class test set, including neutral.
Upstream model selection may already have used official validation; this does
not create a new independent dataset for RoBERTa. No reported complaint or
regression sentence is added to training or used for selecting settings.
See `artifacts/fusion/evaluation.json` for the measured fusion and RoBERTa-alone
comparison on exactly the same test rows.

Reproduce in a staging directory:

```bash
uv run --frozen --extra api python scripts/train_fusion.py --svm-dir model/svm --download --output-dir /tmp/sentilense-fusion-retrain
uv run --frozen --extra api python scripts/install_fusion.py --trained-root /tmp/sentilense-fusion-retrain
```

To retrain the SVM first, use `scripts/train_tweets.py --download --output-dir
/tmp/sentilense-svm-retrain`, then pass its `model/` as `--svm-dir`. Dataset
revisions and checksums are pinned. Preserve the component directory structure
when deploying; include `model/reviews/` and restart after replacing artifacts.

Keyword retrieval uses DuckDuckGo Lite with Bing RSS fallback. It scores only
readable passages matching the complete topic phrase, allows basic plurals and
center/centre spelling, and requires body evidence. Sources contribute equally
to the mean of all three class scores. This describes tone, not author stance
or public opinion. If no readable relevant source is found, no score is produced.
Search snippets are never model input. External search availability can vary.

Explicit URL movie reviews use the complete document and the IMDb SVM, chosen
by review/cinema context. The response identifies `analysis_model` and
`aggregation`; no routing rule forces a sentiment label. Mean class-score page
and keyword summaries are descriptive and have not been separately validated
for webpage accuracy. Sarcasm, other languages and domain shifts can still fail.

Previous logistic and SVM models remain under `artifacts/baseline-*/`.
See [model details](model/README.md), [backend contract](backend/README.md),
and [frontend instructions](frontend/README.md).


The measured three-class test accuracy is **71.43%**, versus
**72.48%** for RoBERTa alone on the same
12,277 tweets. The requested fusion fixes the reported complaint
and passes all 16 regression checks, but is not an overall benchmark improvement.

## Checks

```bash
PYTHONPATH=backend uv run --frozen --extra api pytest backend/tests tests -q
uv run --frozen --extra api python scripts/check_integration.py
cd frontend
npm run build
npm run test:e2e
```

`check_integration.py` expects the complete app to be running. It verifies the
Angular shell, real-model API fidelity, validation, live webpage extraction,
and keyword search through the actual saved model;
its evidence is saved to `artifacts/end_to_end_http.json`.

The frontend browser checks exercise the real backend on desktop and mobile:
the three text examples, live public webpage extraction, live keyword search,
and the reported sarcastic Rediff movie review,
source links, filtering, JSON export, validation, and error recovery.
Reports and screenshots are saved under
`artifacts/`; see [frontend instructions](frontend/README.md).

The trained pipeline and metadata belong in `model/`, and preprocessing must
remain importable from `sentiment_model/`. `SENTILENSE_MODEL_DIR` and
`SENTILENSE_FRONTEND_DIR` support custom absolute deployment paths. Restart
the application after replacing model artifacts or the frontend production build.
Deploy `model/reviews/` too; `SENTILENSE_REVIEW_MODEL_DIR` supports relocating it.
