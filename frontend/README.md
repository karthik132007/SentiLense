# SentiLense frontend

Angular keyword, text, and public-webpage analysis using real FastAPI endpoints.
Text and general pages use a saved SVM + Cardiff Twitter RoBERTa fusion,
with a trained stacking head. Three score bars show negative, neutral and
positive probabilities; neutral is a learned class. Explicit movie reviews
retain the separate binary IMDb SVM and its uncertainty policy.
There are no mock prediction responses. The health indicator requires the
actual `sentiment_model` backend, and inputs are preserved on service failures.

From this directory:

```bash
npm ci
npm run build
```

Then from the repository root:

```bash
uv sync --frozen --extra api
uv run --frozen --extra api python scripts/serve.py
```

Open http://127.0.0.1:8000. Production uses same-origin `/api` calls, and the
server serves `dist/sentiment-intelligence/browser` along with its API.
For frontend development, `npm start` runs at http://127.0.0.1:4200 and targets
the backend at http://127.0.0.1:8000.

Keyword search is the default: enter a topic (for example, `antigravity`), choose
up to three or five sources, and select **Search and analyze**. The interface
identifies the successful search provider: DuckDuckGo is primary, with Bing RSS
as fallback when primary search fails or has no readable relevant sources. It
shows an equal-source summary of matching passage tone, source polarity distribution, linked source cards, topic evidence counts, expandable analyzed topic passages, and skipped/off-topic reasons. Keyword neutral tone is labeled **Neutral**, and no topic score is shown when all sources fail relevance checks. Source filters
and JSON export include the actual search/extraction/model results.

Text and webpage analysis remain available, with the three training smoke-test
examples, readable extraction results, segment filtering, and JSON export.
Neutral is a trained class for fusion outputs. Confidence means the selected
class score. For binary review outputs, neutral still means uncertainty. Responses arriving after a mode switch
are canceled to prevent stale results.

The URL/source display identifies whether it used the SVM + RoBERTa fusion or the
separate IMDb movie-review SVM. Movie-review overall sentiment uses the
complete extracted document; its segment labels are exploratory. Fusion results show all three class bars. Review results show the two binary bars.

With the production build available, run `npm run test:e2e`. The test server
starts automatically if the API isn't already running. Checks exercise the
real models, live keyword search, public URL fetching, and the reported sarcastic
Rediff movie review on desktop and mobile. `topic.spec.ts` separately intercepts keyword responses for deterministic evidence/uncertainty/error UI checks; its fixture uses saved-model scores. These UI fixtures do not validate live retrieval. Run just these checks with `npm run test:e2e -- topic.spec.ts`. Set `SENTILENSE_TEST_URL` to test another running application.
Tests use `/usr/bin/chromium` when installed, or Playwright's installed Chromium;
set `CHROMIUM_PATH` to use another executable. On systems without Chromium,
run `npx playwright install chromium` first.

Reports and screenshots are saved under the repository's `artifacts/`.

The interface uses a neutral workspace shell, black primary controls, muted
sentiment colors, and a compact desktop sidebar. Mobile navigation adapts to
the top bar. Source results use readable lists rather than stacked cards.
New analysis cancels any in-flight request and clears the inputs/results;
input tabs support arrow keys, Home, and End. A skip link, visible focus states,
and reduced-motion styling support keyboard and motion accessibility.

Inter Variable is bundled under `public/fonts/` with its SIL Open Font License;
font loading does not require an external service. Workspace navigation and
the shared polarity display are standalone components. Desktop/mobile browser
checks include keyboard navigation, reset behavior, and font availability;
initial-layout screenshots are `artifacts/frontend-workspace-{desktop,mobile}.png`.
