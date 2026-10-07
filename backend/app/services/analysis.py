"""Business logic for /analyze endpoints."""

import logging
import re

from app.ml.inference import get_adapter, get_review_adapter
from app.scraping.scraper import scrape_url, ScrapedPage, ScrapeError, split_units
from sentiment_model.reviews import is_movie_review

logger = logging.getLogger(__name__)


def analyze_text(text: str) -> dict:
    adapter = get_adapter()
    return adapter.predict(text)


def analyze_url(url: str) -> dict:
    page = scrape_url(url)
    return analyze_page(page)


def analyze_page(page: ScrapedPage, *, topic_units: list[str] | None = None) -> dict:
    """Common real-model analysis for explicitly entered and discovered URLs."""

    units = split_units(page.text) if topic_units is None else topic_units
    if not units:
        raise ScrapeError("Could not extract meaningful content from the page.", 400)

    movie_review = topic_units is None and is_movie_review(page.title, page.text)
    adapter = get_review_adapter() if movie_review else get_adapter()
    results = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    predictions = adapter.predict_many(units)
    for u, pred in zip(units, predictions, strict=True):
        results.append(
            {"text": u, "sentiment": pred["sentiment"],
             "confidence": pred["confidence"], "scores": pred["scores"]}
        )
        counts[pred["sentiment"]] = counts.get(pred["sentiment"], 0) + 1

    total = len(results)
    # Descriptive mean class scores; do not average categorical labels.
    mean_scores = {name: sum(p["scores"][name] for p in predictions) / total
                   for name in predictions[0]["scores"]}
    # Reviews are trained/evaluated as complete documents. Averaging isolated
    # chunks discards the review's framing and dilutes criticism with plot.
    summary = (adapter.predict(page.title + "\n" + page.text) if movie_review else
               adapter.classify_scores(mean_scores))
    overall, confidence = summary["sentiment"], summary["confidence"]

    distribution = {
        k: round(v * 100.0 / total, 1) for k, v in counts.items()
    }
    analyzed_text = page.text if topic_units is None else "\n".join(units)
    words = len(re.findall(r"\S+", analyzed_text))

    return {
        "url": page.url,
        "title": page.title,
        "domain": page.domain,
        "overall_sentiment": overall,
        "confidence": round(confidence, 4),
        "scores": summary["scores"],
        "analysis_model": "imdb_review" if movie_review else adapter.metadata.get("analysis_model", "sentiment140"),
        "aggregation": ("mean_topic_passage_probability" if topic_units is not None else
                        "full_review_document" if movie_review else "mean_segment_probability"),
        "distribution": distribution,
        "statistics": {
            "characters": len(analyzed_text),
            "words": words,
            "analyzed_units": total,
        },
        "results": results,
    }
