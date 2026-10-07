"""Model fidelity, shared thresholds, and webpage integration boundaries."""

import pytest
import httpx
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.ml.inference import get_adapter, RealAdapter
from app.scraping import scraper
from app.scraping.scraper import ScrapeError, ScrapedPage
from app.services import analysis
from sentiment_model import predict_sentiment


@pytest.mark.parametrize("text", ["I absolutely love this!", "I hate this product.",
                                 "The product arrived yesterday."])
def test_api_uses_actual_saved_model(text):
    with TestClient(app) as client:
        response = client.post("/api/analyze/text", json={"text": text})
    assert response.status_code == 200
    assert response.json() == predict_sentiment(text)


def test_reported_negative_text_uses_saved_svm_and_reads_negative():
    text = 'This is fucking trash subject i ever saw in my life, only one with loose brain screw will opt it as minor subject'
    assert get_adapter().metadata['model_family'] == 'svm_roberta_fusion'
    with TestClient(app) as client:
        response = client.post('/api/analyze/text', json={'text': text})
    assert response.status_code == 200
    assert response.json() == predict_sentiment(text)
    assert response.json()['sentiment'] == 'negative'


@pytest.mark.parametrize("position,label", [('lower', 'negative'), ('above_lower', 'neutral'),
                                          ('midpoint', 'neutral'), ('below_upper', 'neutral'),
                                          ('upper', 'positive')])
def test_page_aggregation_uses_saved_thresholds(position, label):
    adapter = RealAdapter(Path(__file__).resolve().parents[2] / 'model/svm')
    lower, upper = adapter.metadata['negative_threshold'], adapter.metadata['positive_threshold']
    probabilities = {'lower': lower, 'above_lower': lower + 1e-6,
                     'midpoint': (lower + upper) / 2, 'below_upper': upper - 1e-6, 'upper': upper}
    assert adapter.classify_probability(probabilities[position])["sentiment"] == label


def test_page_scraping_to_model_contract(monkeypatch):
    text = "I absolutely love this! I hate this product. The product arrived yesterday."
    page = ScrapedPage("https://example.com/review", "A review", "example.com", "", text)
    monkeypatch.setattr(analysis, "scrape_url", lambda url: page)
    with TestClient(app) as client:
        response = client.post("/api/analyze/url", json={"url": page.url})
    assert response.status_code == 200
    result = response.json()
    assert result["statistics"]["analyzed_units"] == len(result["results"])
    for unit in result["results"]:
        expected = predict_sentiment(unit["text"])
        assert unit["sentiment"] == expected["sentiment"]
        assert unit["confidence"] == pytest.approx(expected["confidence"], abs=1e-5)
    assert sum(result["distribution"].values()) == pytest.approx(100, abs=0.2)


def test_model_missing_never_uses_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTILENSE_MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        RealAdapter()


def test_control_only_text_is_validation_error():
    with TestClient(app) as client:
        response = client.post("/api/analyze/text", json={"text": "\x00\ufffd"})
    assert response.status_code == 422


def test_redirect_to_localhost_is_blocked_before_request(monkeypatch):
    requested = []

    def handle(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    original_client = httpx.Client
    monkeypatch.setattr(scraper, "_check_dns", lambda host: None)
    monkeypatch.setattr(scraper.httpx, "Client", lambda **kwargs:
                        original_client(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(ScrapeError, match="not allowed"):
        scraper.fetch_html("https://example.com/article")
    assert requested == ["https://example.com/article"]


def test_extract_nested_navigation_classes():
    html = '<html><body><div class="menu"><div class="menu-item">Boilerplate</div></div><main><p>I absolutely love this wonderful product and its amazing features!</p></main></body></html>'
    page = scraper.extract_content(html, "https://example.com/article")
    assert "Boilerplate" not in page.text
    assert "wonderful product" in page.text
