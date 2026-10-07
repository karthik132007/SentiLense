#!/usr/bin/env python3
"""Verify a running frontend/API/model application without mocking responses."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from sentiment_model import predict_sentiment
from sentiment_model.reviews import predict_review, review_model_dir


def verify_page(page):
    review = page['analysis_model'] == 'imdb_review'
    predict = predict_review if review else predict_sentiment
    probabilities = []
    for unit in page['results']:
        expected = predict(unit['text'])
        assert unit['sentiment'] == expected['sentiment']
        assert abs(unit['confidence'] - expected['confidence']) < 1e-5
        assert unit['scores'].keys() == expected['scores'].keys()
        assert all(abs(unit['scores'][k] - expected['scores'][k]) < 1e-5 for k in expected['scores'])
        probabilities.append(expected['scores']['positive'])
    assert probabilities and page['statistics']['analyzed_units'] == len(probabilities)
    if review:
        # Full-review scores are not the mean of isolated plot segments.
        sys.path.insert(0, str(ROOT / 'backend'))
        from app.scraping.scraper import scrape_url
        content = scrape_url(page['url'])
        expected = predict_review(content.title + '\n' + content.text)
        assert page['scores'] == expected['scores']
        assert page['overall_sentiment'] == expected['sentiment']
    else:
        assert abs(page['scores']['positive'] - sum(probabilities) / len(probabilities)) < 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--url", default="https://example.com", help="Public webpage to exercise actual extraction/inference")
    parser.add_argument("--keyword", default="antigravity", help="Keyword to exercise live search, extraction, and inference")
    parser.add_argument("--allow-unbuilt-frontend", action="store_true", help="API-only verification while the frontend is still being built")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "base_url": base,
              "mocked_responses": False, "frontend_verified": False}
    with httpx.Client(timeout=180) as client:
        health = client.get(base + "/api/health")
        health.raise_for_status()
        assert health.json() == {"status": "ok", "model_loaded": True, "model_backend": "sentiment_model"}
        report["health"] = health.json()
        root = client.get(base + "/")
        root.raise_for_status()
        if not args.allow_unbuilt_frontend:
            assert "text/html" in root.headers.get("content-type", ""), "Frontend has not been built/served"
            assert "<app-root" in root.text, "Expected the Angular application shell"
            report["frontend_verified"] = True
        examples = []
        for text in ["I absolutely love this!", "I hate this product.", "The product arrived yesterday."]:
            response = client.post(base + "/api/analyze/text", json={"text": text})
            response.raise_for_status()
            assert response.json() == predict_sentiment(text), "API differs from saved-model inference"
            examples.append({"text": text, "prediction": response.json()})
        report["text_examples"] = examples
        assert client.post(base + "/api/analyze/text", json={"text": "   "}).status_code == 422
        assert client.post(base + "/api/analyze/url", json={"url": "http://127.0.0.1/private"}).status_code == 400
        assert client.get(base + "/api/not-a-route").status_code == 404
        assert client.get(base + "/missing-asset.js").status_code == 404
        report["validation_and_missing_route_checks_passed"] = True
        response = client.post(base + "/api/analyze/url", json={"url": args.url})
        response.raise_for_status()
        page = response.json()
        assert page["title"] and page["domain"] and page["results"]
        assert page["statistics"]["analyzed_units"] == len(page["results"])
        assert abs(sum(page["distribution"].values()) - 100) <= 0.2
        verify_page(page)
        model_dir = Path(os.environ.get("SENTILENSE_MODEL_DIR", ROOT / "model"))
        metadata = json.loads((model_dir / "model_metadata.json").read_text())
        page_metadata_dir = review_model_dir() if page['analysis_model'] == 'imdb_review' else model_dir
        page_metadata = json.loads((page_metadata_dir / 'model_metadata.json').read_text())
        from sentiment_model.fusion import score_result
        mean_positive = page['scores']['positive']
        expected_label = (score_result(page['scores'])['sentiment'] if 'neutral' in page['scores'] else
                          "positive" if mean_positive >= page_metadata["positive_threshold"] else
                          "negative" if mean_positive <= page_metadata["negative_threshold"] else "neutral")
        assert page["overall_sentiment"] == expected_label
        report["live_webpage"] = page
        response = client.post(base + "/api/analyze/keyword", json={"keyword": args.keyword, "max_sources": 5})
        response.raise_for_status()
        topic = response.json()
        assert topic["sources"] and topic["keyword"] == args.keyword
        assert topic["statistics"]["sources_analyzed"] == len(topic["sources"]) <= 5
        assert topic["statistics"]["sources_skipped"] == len(topic["skipped_sources"])
        assert len({source["url"] for source in topic["sources"]}) == len(topic["sources"])
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for source in topic["sources"]:
            verify_page(source)
            counts[source["overall_sentiment"]] += 1
        mean_positive = sum(source["scores"]["positive"] for source in topic["sources"]) / len(topic["sources"])
        assert abs(topic["scores"]["positive"] - mean_positive) < 1e-12
        expected_label = (score_result(topic['scores'])['sentiment'] if 'neutral' in topic['scores'] else
                          "positive" if mean_positive >= metadata["positive_threshold"] else
                          "negative" if mean_positive <= metadata["negative_threshold"] else "neutral")
        assert topic["overall_sentiment"] == expected_label
        assert topic["distribution"] == {label: round(count * 100 / len(topic["sources"]), 1) for label, count in counts.items()}
        report["live_keyword"] = topic
    report["passed"] = True
    path = ROOT / "artifacts/end_to_end_http.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "evidence": str(path), "keyword": args.keyword,
                      "keyword_statistics": topic["statistics"],
                      "overall_sentiment": topic["overall_sentiment"]}, indent=2))


if __name__ == "__main__":
    main()
