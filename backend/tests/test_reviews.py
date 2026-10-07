"""Document routing and extraction regressions using the actual saved models."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.scraping.scraper import ScrapedPage, extract_content
from app.services import analysis
from sentiment_model.reviews import is_movie_review, predict_review


@pytest.mark.parametrize('title,text,expected', [
    ('The Example Movie Review', 'A film about a family.', True),
    ('The Example Review: A Disappointment', 'The movie has poor acting and the director wastes its screenplay.', True),
    ('Coffee Machine Review', 'A coffee machine for the kitchen.', False),
    ('Film festival announces directors', 'An actor introduces a new movie.', False),
])
def test_domain_routing_does_not_depend_on_sentiment(title, text, expected):
    assert is_movie_review(title, text) == expected


@pytest.mark.parametrize('text,label', [
    ('The film is a tedious mess. The script makes no sense. The acting is wooden, the direction is terrible, and I regretted watching it.', 'negative'),
    ('An excellent film with a brilliant screenplay and wonderful performances. I loved every minute. The director created a masterpiece that I would happily watch again.', 'positive'),
])
def test_movie_review_uses_full_document_model(monkeypatch, text, label):
    page = ScrapedPage('https://example.com/review', 'Example Movie Review', 'example.com', '', text)
    monkeypatch.setattr(analysis, 'scrape_url', lambda url: page)
    with TestClient(app) as client:
        response = client.post('/api/analyze/url', json={'url': page.url})
    assert response.status_code == 200
    result = response.json()
    expected = predict_review(page.title + '\n' + page.text)
    assert result['overall_sentiment'] == expected['sentiment'] == label
    assert result['scores'] == expected['scores']
    assert result['confidence'] == round(expected['confidence'], 4)
    assert result['analysis_model'] == 'imdb_review'
    assert result['aggregation'] == 'full_review_document'
    for unit in result['results']:
        assert unit['scores'] == predict_review(unit['text'])['scores']


def test_news_article_excludes_unrelated_service_recommendations():
    paragraphs = [
        'This film is a tiresome exercise in spectacle, with a screenplay that never explains the motivation of its characters. The reviewer found the direction incoherent and the performances disappointing.',
        'The movie spends so much time introducing subplots that it forgets to tell a compelling story. The jokes are forced, the pacing is tedious, and the finale offers no satisfying resolution.',
        'The verdict is that this is a deeply disappointing movie. Neither the director nor the actors can rescue the confused screenplay, and the audience is left wondering why it was made.',
    ]
    links = ''.join(f'<a href="/unrelated/{i}">Recommended products and shopping services {i}</a><br>' for i in range(70))
    html = '<html><head><title>Example Movie Review</title></head><body><div class="story-text">' + ''.join('<p>' + paragraph + '</p>' for paragraph in paragraphs) + '</div><div class="related-stories">' + links + '</div></body></html>'
    page = extract_content(html, 'https://example.com/review')
    assert 'disappointing movie' in page.text
    assert 'shopping services' not in page.text
