"""Backend tests: health, text analysis, URL validation, extraction."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.scraping import scraper
from app.scraping.scraper import ScrapeError, extract_content, split_units, validate_url

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_backend"] == "sentiment_model"


def test_text_analysis_positive():
    r = client.post("/api/analyze/text", json={"text": "I absolutely love this!"})
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "positive" in data["scores"] and "negative" in data["scores"]


def test_text_analysis_negative():
    r = client.post("/api/analyze/text", json={"text": "I hate this terrible awful thing"})
    assert r.status_code == 200
    assert r.json()["sentiment"] == "negative"


def test_empty_text_rejected():
    r = client.post("/api/analyze/text", json={"text": "   "})
    assert r.status_code == 422
    r2 = client.post("/api/analyze/text", json={"text": ""})
    assert r2.status_code == 422


def test_url_validation_bad_scheme():
    with pytest.raises(ScrapeError):
        validate_url("ftp://example.com/file")


def test_invalid_url_endpoint():
    r = client.post("/api/analyze/url", json={"url": "not-a-url"})
    assert r.status_code in (400, 422)


def test_ssrf_localhost_blocked():
    for bad in [
        "http://localhost/article",
        "http://127.0.0.1/article",
        "http://10.0.0.5/article",
        "http://192.168.1.1/article",
        "http://169.254.169.254/latest",
    ]:
        r = client.post("/api/analyze/url", json={"url": bad})
        assert r.status_code in (400, 422), bad


def test_unreachable_website():
    # Reserved TEST-NET / invalid TLD should fail gracefully, not 500-crash
    r = client.post(
        "/api/analyze/url", json={"url": "http://example.invalid/article"}
    )
    assert r.status_code in (400, 422, 500)
    assert "detail" in r.json()


def test_html_extraction():
    html = """
    <html><head><title>Test Article</title>
    <meta name="description" content="A short desc."></head>
    <body><nav>Home About</nav>
    <script>var x = 1;</script>
    <style>.a{color:red}</style>
    <main><article>
    <p>I love this wonderful product. It works great!</p>
    <p>But the delivery was terrible and slow.</p>
    </article></main>
    <footer>copyright</footer></body></html>
    """
    page = extract_content(html, "https://example.com/article")
    assert page.title == "Test Article"
    assert page.domain == "example.com"
    assert page.description == "A short desc."
    assert "love this wonderful" in page.text
    assert "var x" not in page.text
    assert "Home About" not in page.text


def test_split_units():
    text = "I love this. " * 50 + "It is terrible. " * 10
    units = split_units(text)
    assert 1 <= len(units) <= 100
    assert all(len(u) <= 600 for u in units)


def test_webpage_analysis_with_mocked_fetch(monkeypatch):
    html = """
    <html><head><title>Mock Page</title></head><body><main>
    <p>I love this amazing wonderful product! It is fantastic.</p>
    <p>This is a plain statement of fact with no feeling words here today.</p>
    <p>I hate this terrible awful horrible product. It is the worst.</p>
    </main></body></html>
    """

    def fake_fetch(url):
        return html, url

    monkeypatch.setattr(scraper, "fetch_html", fake_fetch)
    monkeypatch.setattr(scraper, "_check_dns", lambda host: None)
    r = client.post("/api/analyze/url", json={"url": "https://example.com/mock"})
    assert r.status_code == 200
    data = r.json()
    assert data["url"] == "https://example.com/mock"
    assert data["title"] == "Mock Page"
    assert data["domain"] == "example.com"
    assert data["overall_sentiment"] in ("positive", "neutral", "negative")
    assert isinstance(data["confidence"], float)
    assert set(data["distribution"].keys()) == {"positive", "neutral", "negative"}
    assert data["statistics"]["analyzed_units"] == len(data["results"])
    assert data["statistics"]["words"] > 0
    assert all(
        {"text", "sentiment", "confidence"}.issubset(item.keys())
        for item in data["results"]
    )
