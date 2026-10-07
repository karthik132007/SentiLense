"""Keyword -> ranked search results -> scraped pages -> real model outputs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import logging

from app.ml.inference import get_adapter
from app.scraping.scraper import ScrapeError, scrape_url
from app.scraping.search import SEARCH_PROVIDERS, SearchResult, search_web
from app.scraping.relevance import topic_passages
from app.services.analysis import analyze_page
from sentiment_model.preprocessing import normalize_text

logger = logging.getLogger(__name__)


def _read_source(result: SearchResult, keyword: str):
    try:
        page = scrape_url(result.url)  # Includes URL/DNS/redirect checks.
        fingerprint = hashlib.sha256(normalize_text(page.text).encode()).hexdigest()
        units, evidence = topic_passages(page, keyword)
        source = analyze_page(page, topic_units=units)
        source["topic_evidence"] = evidence
        return source, fingerprint, None
    except ScrapeError as exc:
        return None, None, str(exc)
    except Exception:
        logger.exception("Keyword source analysis failed")
        return None, None, "Could not analyze this source."


def analyze_keyword(keyword: str, max_sources: int = 5) -> dict:
    sources, skipped = [], []
    fingerprints, final_urls = set(), set()
    attempted_urls = set()
    search_results_found = 0
    search_errors = []
    # Batches preserve search ranking, limit parallel fetches to three, and stop
    # once enough readable sources have been analyzed.
    with ThreadPoolExecutor(max_workers=3) as pool:
        for engine in SEARCH_PROVIDERS:
            try:
                candidates, provider = search_web(keyword, limit=15, provider=engine)
            except ScrapeError as exc:
                search_errors.append(exc)
                logger.warning("%s search failed: %s", engine, exc)
                continue
            search_results_found += len(candidates)
            # Do not refetch an already rejected URL from the fallback provider.
            candidates = [candidate for candidate in candidates if candidate.url not in attempted_urls]
            position = 0
            while position < len(candidates) and len(sources) < max_sources:
                batch_size = min(3, max_sources - len(sources))
                batch = candidates[position:position + batch_size]
                position += len(batch)
                attempted_urls.update(candidate.url for candidate in batch)
                for candidate, (source, fingerprint, failure) in zip(batch, pool.map(_read_source, batch, [keyword] * len(batch))):
                    if failure:
                        skipped.append({"url": candidate.url, "title": candidate.title, "reason": failure})
                        continue
                    if fingerprint in fingerprints or source["url"] in final_urls:
                        skipped.append({"url": candidate.url, "title": candidate.title, "reason": "Duplicate source content."})
                        continue
                    fingerprints.add(fingerprint)
                    final_urls.add(source["url"])
                    sources.append(source)
            # Use fallback when the primary fails or yields no readable,
            # relevant pages. Keep usable partial results from one provider.
            if sources:
                break
    if not sources:
        if not search_results_found and search_errors:
            raise next((error for error in search_errors if error.status_code == 503), search_errors[-1])
        raise ScrapeError("No readable webpages with sufficient content matching this topic were found. No topic score was produced. Try another phrase or analyze a specific URL.", 422)
    # Equal source weighting prevents a single long page dominating the topic.
    mean_scores = {name: sum(source["scores"][name] for source in sources) / len(sources)
                   for name in sources[0]["scores"]}
    summary = get_adapter().classify_scores(mean_scores)
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for source in sources:
        counts[source["overall_sentiment"]] += 1
    return {
        "keyword": keyword, "search_provider": provider,
        "searched_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_sentiment": summary["sentiment"], "confidence": summary["confidence"],
        "scores": summary["scores"],
        "distribution": {label: round(count * 100 / len(sources), 1) for label, count in counts.items()},
        "statistics": {
            "search_results_found": search_results_found, "sources_analyzed": len(sources),
            "sources_skipped": len(skipped), "requested_sources": max_sources,
            "analyzed_units": sum(source["statistics"]["analyzed_units"] for source in sources),
            "words": sum(source["statistics"]["words"] for source in sources),
            "characters": sum(source["statistics"]["characters"] for source in sources),
        },
        "aggregation": "Equal mean of each source's topic-passage class scores; distribution counts source tone labels. Tone does not establish topic stance or public opinion.",
        "sources": sources, "skipped_sources": skipped,
    }
