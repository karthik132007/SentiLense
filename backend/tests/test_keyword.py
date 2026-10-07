"""Search/extraction fixtures for failure boundaries; real saved model inference."""

from xml.sax.saxutils import escape

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.inference import get_adapter
from app.scraping import search
from app.scraping.scraper import ScrapedPage, ScrapeError
from app.scraping.search import SearchResult, parse_results, parse_duckduckgo_results
from app.scraping.relevance import mentions_topic, topic_passages
from app.services import keyword_analysis as service
from sentiment_model import predict_sentiment


def feed(links):
    return ('<rss><channel>' + ''.join(
        f'<item><title>Source {i}</title><link>{escape(link)}</link></item>'
        for i, link in enumerate(links)) + '</channel></rss>').encode()


def test_search_parser_preserves_ranking_deduplicates_and_rejects_bad_links():
    results = parse_results(feed(['https://example.com/a#one', 'https://example.com/a#two',
                                  'javascript:alert(1)', 'https://user:pass@example.com',
                                  'https://[broken', 'https://example.org/b',
                                  'https://example.net/c']), 2)
    assert [item.url for item in results] == ['https://example.com/a', 'https://example.org/b']


@pytest.mark.parametrize('xml', [b'<html>Blocked</html>', b'<rss><broken>',
                                b'<!DOCTYPE rss [<!ENTITY secret SYSTEM "file:///etc/passwd">]><rss><channel><item><link>&secret;</link></item></channel></rss>'])
def test_search_parser_does_not_accept_bad_feeds_or_expand_entities(xml):
    try:
        assert parse_results(xml, 5) == []
    except ScrapeError as exc:
        assert exc.status_code == 503


@pytest.mark.parametrize('response,status', [(httpx.Response(200, content=feed([])), 404),
                                            (httpx.Response(429), 503)])
def test_search_empty_and_unavailable_are_explicit(monkeypatch, response, status):
    original_client = httpx.Client
    requests = []
    def handle(request):
        requests.append(request)
        return response
    monkeypatch.setattr(search.httpx, 'Client', lambda **kwargs:
                        original_client(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(ScrapeError) as exc:
        search.search_web('antigravity', provider='Bing')
    assert exc.value.status_code == status
    assert requests[0].url.params['q'] == 'antigravity'
    assert requests[0].url.params['format'] == 'rss'


def install_sources(monkeypatch):
    urls = ['https://example.com/' + name for name in ['blocked', 'positive', 'duplicate', 'negative', 'unused']]
    candidates = [SearchResult(url.rsplit('/', 1)[1], url) for url in urls]
    calls = []
    def fake_scrape(url):
        calls.append(url)
        name = url.rsplit('/', 1)[1]
        if name == 'blocked':
            raise ScrapeError('HTTP 403 from webpage.', 400)
        text = ('I absolutely love this wonderful test topic! It is amazing. ' * 12
                if name in {'positive', 'duplicate'} else 'I hate this terrible test topic.')
        return ScrapedPage(url, name, 'example.com', '', text)
    monkeypatch.setattr(service, 'search_web', lambda keyword, limit, **kwargs: (candidates, 'Bing'))
    monkeypatch.setattr(service, 'scrape_url', fake_scrape)
    return calls


def test_keyword_partial_failures_deduplication_and_equal_source_weighting(monkeypatch):
    calls = install_sources(monkeypatch)
    with TestClient(app) as client:
        response = client.post('/api/analyze/keyword', json={'keyword': '  test   topic  ', 'max_sources': 2})
    assert response.status_code == 200
    topic = response.json()
    assert topic['keyword'] == 'test topic'
    assert [source['title'] for source in topic['sources']] == ['positive', 'negative']
    assert topic['statistics']['sources_skipped'] == 2
    assert [item['reason'] for item in topic['skipped_sources']] == ['HTTP 403 from webpage.', 'Duplicate source content.']
    assert not any(url.endswith('unused') for url in calls)
    for source in topic['sources']:
        for unit in source['results']:
            expected = predict_sentiment(unit['text'])
            assert unit['sentiment'] == expected['sentiment']
            assert unit['scores'] == pytest.approx(expected['scores'], abs=1e-5)
            assert unit['confidence'] == pytest.approx(expected['confidence'], abs=1e-5)
    mean_scores = {name: sum(source['scores'][name] for source in topic['sources']) / 2
                   for name in topic['sources'][0]['scores']}
    summary = get_adapter().classify_scores(mean_scores)
    assert topic['scores'] == summary['scores']
    assert topic['overall_sentiment'] == summary['sentiment']
    assert topic['distribution'] == {'positive': 50.0, 'negative': 50.0, 'neutral': 0.0}
    assert topic['statistics']['analyzed_units'] == sum(len(source['results']) for source in topic['sources'])
    assert all(mentions_topic(unit['text'], topic['keyword']) for source in topic['sources'] for unit in source['results'])


def test_no_readable_sources_returns_actionable_error(monkeypatch):
    monkeypatch.setattr(service, 'search_web', lambda keyword, limit, **kwargs:
                        ([SearchResult('Private', 'http://127.0.0.1/private')], 'Bing'))
    # Exercise the actual scraper URL check, without a private network request.
    with TestClient(app) as client:
        response = client.post('/api/analyze/keyword', json={'keyword': 'private'})
    assert response.status_code == 422
    assert 'No readable webpages' in response.json()['detail']


def test_multiword_search_preserves_the_complete_phrase(monkeypatch):
    original_client = httpx.Client
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, content=feed(['https://example.com/data-centers']))
    monkeypatch.setattr(search.httpx, 'Client', lambda **kwargs:
                        original_client(transport=httpx.MockTransport(handle), **kwargs))
    search.search_web('Data centers', provider='Bing')
    assert requests[0].url.params['q'] == '"Data centers"'


def test_duckduckgo_parser_unwraps_ranked_links_without_using_snippets():
    html = b'''<html><body><table>
      <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fcenter%23part&amp;rut=unused">Data center</a>
      <td class="result-snippet">I absolutely love this wonderful thing!</td>
      <a class="result-link" href="https://example.com/center#other">Duplicate</a>
      <a class="result-link" href="javascript:alert(1)">Invalid</a>
      <a class="result-link" href="https://user:pass@example.com/secret">Credentials</a>
      <a class="result-link" href="http://[broken">Malformed</a>
      <a class="result-link" href="https://example.org/data-center">Another center</a>
      <a href="https://example.net/ad">Unrelated navigation/ad</a>
    </table></body></html>'''
    assert parse_duckduckgo_results(html, 5) == [
        SearchResult('Data center', 'https://example.com/center'),
        SearchResult('Another center', 'https://example.org/data-center'),
    ]
    assert parse_duckduckgo_results(html, 1) == [SearchResult('Data center', 'https://example.com/center')]


@pytest.mark.parametrize('html', [
    b'<html><form id="challenge-form">Verify you are human</form></html>',
    b'<html><p>Temporarily blocked</p></html>',
])
def test_duckduckgo_challenges_and_invalid_responses_are_provider_failures(html):
    with pytest.raises(ScrapeError) as exc:
        parse_duckduckgo_results(html, 5)
    assert exc.value.status_code == 503


def test_default_search_uses_duckduckgo_and_keeps_all_query_words(monkeypatch):
    requests = []
    original_client = httpx.Client
    def handle(request):
        requests.append(request)
        return httpx.Response(200, content=b'<a class="result-link" href="https://example.com/data-centers">Data centers</a>')
    monkeypatch.setattr(search.httpx, 'Client', lambda **kwargs:
                        original_client(transport=httpx.MockTransport(handle), **kwargs))
    results, provider = search.search_web('Data centers')
    assert provider == 'DuckDuckGo'
    assert requests[0].url.host == 'lite.duckduckgo.com'
    assert requests[0].url.params['q'] == 'Data centers'
    assert results == [SearchResult('Data centers', 'https://example.com/data-centers')]


@pytest.mark.parametrize('primary_failure', ['unavailable', 'off-topic', 'unreadable'])
def test_fallback_provider_is_checked_when_primary_has_no_usable_sources(monkeypatch, primary_failure):
    searches, scrapes = [], []
    def fake_search(keyword, limit, *, provider):
        searches.append(provider)
        if provider == 'DuckDuckGo' and primary_failure == 'unavailable':
            raise ScrapeError('Search unavailable', 503)
        name = 'wrong' if provider == 'DuckDuckGo' else 'right'
        return [SearchResult(name, f'https://example.com/{name}')], provider
    def fake_scrape(url):
        scrapes.append(url)
        if url.endswith('/wrong'):
            if primary_failure == 'unreadable':
                raise ScrapeError('Website returned HTTP 403.', 400)
            return ScrapedPage(url, 'What is Data?', 'example.com', '', 'Data is a collection of facts and observations.')
        return ScrapedPage(url, 'Data centers', 'example.com', '', 'The data centers were built yesterday.')
    monkeypatch.setattr(service, 'search_web', fake_search)
    monkeypatch.setattr(service, 'scrape_url', fake_scrape)
    result = service.analyze_keyword('data centers', 1)
    assert searches == ['DuckDuckGo', 'Bing']
    assert result['search_provider'] == 'Bing'
    assert [source['url'] for source in result['sources']] == ['https://example.com/right']
    assert result['statistics']['search_results_found'] == (1 if primary_failure == 'unavailable' else 2)


def test_readable_primary_does_not_request_fallback(monkeypatch):
    searches = []
    def fake_search(keyword, limit, *, provider):
        searches.append(provider)
        return [SearchResult('Data center', 'https://example.com/center')], provider
    monkeypatch.setattr(service, 'search_web', fake_search)
    monkeypatch.setattr(service, 'scrape_url', lambda url: ScrapedPage(
        url, 'Data center', 'example.com', '', 'The data center was built yesterday.'))
    result = service.analyze_keyword('Data centers', 5)
    assert searches == ['DuckDuckGo']
    assert result['search_provider'] == 'DuckDuckGo'
    assert result['statistics']['sources_analyzed'] == 1


def test_all_search_providers_unavailable_return_503(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ScrapeError('Web search is temporarily unavailable.', 503)
    monkeypatch.setattr(service, 'search_web', unavailable)
    with TestClient(app) as client:
        response = client.post('/api/analyze/keyword', json={'keyword': 'data centers'})
    assert response.status_code == 503


@pytest.mark.parametrize('text,expected', [
    ('A data center is a facility.', True),
    ('New DATA-CENTRES are being built.', True),
    ('Data describes facts. Shopping centers are popular.', False),
    ('Data is a collection of values.', False),
    ('The database centers on statistics.', False),
])
def test_topic_matching_does_not_reduce_a_phrase_to_one_word(text, expected):
    assert mentions_topic(text, 'Data centers') == expected


def test_reported_data_definition_results_produce_no_topic_verdict(monkeypatch):
    candidates = [SearchResult(title, f'https://example.com/{i}') for i, title in enumerate([
        'Data - Wikipedia', 'What is Data? | IBM', 'Data and its Types',
        'World Bank Open Data', 'What is Data? Definition, Types, and Examples',
    ])]
    monkeypatch.setattr(service, 'search_web', lambda keyword, limit, **kwargs: (candidates, 'Bing'))
    monkeypatch.setattr(service, 'scrape_url', lambda url: ScrapedPage(
        url, 'What is Data?', 'example.com', '',
        'Data is a collection of facts, numbers, words and observations. It helps organizations make decisions.'))
    def unexpected_analysis(*args, **kwargs):
        pytest.fail('Off-topic content must be rejected before sentiment inference')
    monkeypatch.setattr(service, 'analyze_page', unexpected_analysis)
    with TestClient(app) as client:
        response = client.post('/api/analyze/keyword', json={'keyword': 'Data centers'})
    assert response.status_code == 422
    assert 'No topic score was produced' in response.json()['detail']


def test_topic_passages_exclude_unrelated_tone_and_find_late_evidence():
    unrelated = ('I love this amazing wonderful product!\n' * 150)
    page = ScrapedPage('https://example.com/article', 'Data center risks', 'example.com', '',
                       unrelated + 'Data centers are causing terrible noise and I hate the disruption.')
    units, evidence = topic_passages(page, 'data centers')
    assert units == ['Data centers are causing terrible noise and I hate the disruption.']
    assert evidence['matched_passages'] == 1
    assert evidence['extracted_passages'] == 151
    result = service.analyze_page(page, topic_units=units)
    assert result['scores'] == predict_sentiment(units[0])['scores']
    assert result['statistics']['words'] == len(units[0].split())
    assert result['aggregation'] == 'mean_topic_passage_probability'


def test_lowercase_sentences_do_not_carry_unrelated_polarity_into_topic_passages():
    page = ScrapedPage('https://example.com/topic', 'Data centers', 'example.com', '',
                       'I love this fantastic product! data centers store files. I hate this horrible movie.')
    units, _ = topic_passages(page, 'Data centers')
    assert units == ['data centers store files.']


@pytest.mark.parametrize('title,text', [
    ('What is Data?', 'Data is useful.\nData centers store some data.\nData includes facts.\nTypes include numbers.\nStatistics summarize observations.\nValues are collected.'),
    ('Data centers', 'Data is a collection of facts and observations.'),
])
def test_incidental_mentions_and_search_titles_without_body_evidence_are_rejected(title, text):
    with pytest.raises(ScrapeError, match='Off-topic'):
        topic_passages(ScrapedPage('https://example.com/data', title, 'example.com', '', text), 'Data centers')


@pytest.mark.parametrize('body', [{'keyword': ''}, {'keyword': '   '}, {'keyword': '\x00'},
                                {'keyword': 'x' * 201}, {'keyword': 'test', 'max_sources': 0},
                                {'keyword': 'test', 'max_sources': 6},
                                {'keyword': 'test', 'max_sources': True}])
def test_keyword_validation_never_searches_invalid_requests(monkeypatch, body):
    def unexpected_search(*args, **kwargs):
        pytest.fail('Search should not run for invalid input')
    monkeypatch.setattr(service, 'search_web', unexpected_search)
    with TestClient(app) as client:
        assert client.post('/api/analyze/keyword', json=body).status_code == 422
