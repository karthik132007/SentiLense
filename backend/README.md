# SentiLense Backend

FastAPI backend + web scraping engine for the SentiLense academic
sentiment analysis platform.

## Run

From the repository root:

```bash
uv sync --frozen --extra api
uv run --frozen --extra api python scripts/serve.py --reload
```

API docs: http://127.0.0.1:8000/docs

## Endpoints (frontend contract)

- `GET /api/health` → `{"status": "ok", "model_loaded": true,
  "model_backend": "sentiment_model"}` when ready
- `POST /api/analyze/text` → `{sentiment, confidence, scores}`
- `POST /api/analyze/url` → `{url, title, domain, overall_sentiment,
  confidence, scores, distribution, statistics, results, analysis_model, aggregation}`
- `POST /api/analyze/keyword` accepts `{"keyword":"antigravity","max_sources":5}`
  and returns `{keyword, search_provider, searched_at_utc, overall_sentiment,
  confidence, scores, distribution, statistics, aggregation, sources, skipped_sources}`.
  Each `sources` item contains topic-passage scores and `topic_evidence` (matching method, title match, matched/extracted passage counts, and extracted word count).

Keywords must be nonblank strings of at most 200 characters; `max_sources` is
an integer from 1 to 5, defaulting to 5. Ranked URLs are discovered through DuckDuckGo's
public Lite search. If it fails or produces no readable relevant sources, Bing RSS
is tried with a quoted multiword query. Usable partial primary results are retained
without mixing providers. `search_provider` names the provider supplying included
sources; `search_results_found` counts results across attempted providers.
DuckDuckGo tracking links are decoded into destination URLs, then pages are fetched
with the existing URL/DNS/redirect protection. Search challenges are treated as
provider failures and are never submitted or solved.
No API key or new dependency is required. Search snippets are never classified.
Duplicate normalized page content and redirected URLs are omitted. Fetches run
in batches with at most three workers and stop when enough readable sources
have been found. Extracted content must match the complete topic phrase, allowing basic English plurals and center/centre spelling. Title evidence plus at least one body passage, or repeated/substantial body evidence, is required; search titles alone cannot qualify a page. This lexical gate does not resolve synonyms or entity meanings. Failed/off-topic sources include reasons; all unreadable or off-topic returns 422 without a score,
no search results returns 404, and provider failure returns 503.

Keyword tone applies argmax to the mean class scores of matching topic passages, weighting each source equally. Unrelated sentences are excluded before inference; the full document is searched before limiting inference to 100 matching units. Keyword sources always use the fusion model and report `mean_topic_passage_probability`; explicit URL movie reviews retain document routing. Its distribution counts source
labels, whereas URL distribution counts segment labels. Keyword source statistics count scored passage content; `topic_evidence.extracted_words` records the full extraction size. The interface displays the trained `neutral` class as **Neutral**. Passage tone is not topic stance, and factual text may be misclassified by this tweet model.
Search results can change and may cover multiple meanings of a keyword.

CORS is enabled for Angular dev servers on `localhost:4200` / `127.0.0.1:4200`.

## Model integration

Text/general webpages/keyword passages use the saved SVM + Cardiff Twitter
RoBERTa fusion. A learned calibrated stacking head combines SVM log odds and
three RoBERTa log scores. It returns negative/neutral/positive probabilities;
neutral is a trained class and confidence is the winning score. The API's
`scores` dictionary now includes `neutral` for fusion outputs. Class labels and
endpoint shapes remain the same. `analysis_model` is `svm_roberta_fusion`.

Startup validates all expert/head checksums. Missing/corrupt components stop
startup without alternate or mock predictions. RoBERTa is loaded entirely from
local safetensors with no runtime Hugging Face calls. Page inference batches
segments to reduce CPU overhead. SVM and RoBERTa stay frozen; the head uses
separate train/calibration/selection subsets of official TweetEval validation.
The cleaned official three-class test is never used to fit or select the head.

General pages average every class probability across segments and select the
largest. Keyword mode first selects matching topic passages, then averages
source class scores with equal source weight. Mean scores describe extracted
tone; tweet metrics do not establish webpage accuracy, stance or public opinion.

Explicit movie-review pages retain the separate binary IMDb LinearSVC on the
complete extracted title/document, reporting `imdb_review` and
`full_review_document`. Its neutral label means uncertainty; scores contain
only positive/negative. Review segments remain exploratory. Review/cinema
context selects this expert without forcing its polarity.

Set `SENTILENSE_MODEL_DIR` for the complete fusion artifact directory and
`SENTILENSE_REVIEW_MODEL_DIR` for `model/reviews/`. Both are warmed at startup.

Article extraction prefers a meaningful cleaned main/article container, then
trafilatura's article text, then cleaned body text. It no longer selects the
longest fallback, which included unrelated recommendations and service links.

Build the Angular frontend in `frontend/` with `npm ci && npm run build`.
The server then serves its production build and `/api` from the same origin at
http://127.0.0.1:8000. Set `SENTILENSE_FRONTEND_DIR` for a custom build location.
Missing API endpoints/assets return 404; Angular navigation gets the app shell.

## Tests

```bash
PYTHONPATH=backend uv run --frozen --extra api pytest backend/tests tests -q
```
