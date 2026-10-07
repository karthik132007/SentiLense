"""Discover public webpages; extracted content is independently checked later."""

from dataclasses import dataclass
from urllib.parse import parse_qs, urldefrag, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from lxml import etree

from app.scraping.scraper import ScrapeError, USER_AGENT, MAX_BYTES

SEARCH_URL = "https://www.bing.com/search"
DUCKDUCKGO_URL = "https://lite.duckduckgo.com/lite/"
SEARCH_PROVIDERS = ("DuckDuckGo", "Bing")
# The public HTML search interface expects a browser-compatible client.
# Challenges are reported as unavailable; they are never solved or submitted.
SEARCH_USER_AGENT = "Mozilla/5.0"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str


def _public_link(raw_url: str) -> str | None:
    try:
        url = urldefrag(raw_url.strip())[0]
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return None
        parsed.port
        return url
    except ValueError:
        return None


def parse_results(xml: bytes, limit: int) -> list[SearchResult]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(xml, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ScrapeError("Web search is temporarily unavailable. Please try again.", 503) from exc
    if root.tag != "rss":
        raise ScrapeError("Web search is temporarily unavailable. Please try again.", 503)
    results, seen_urls = [], set()
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        url = _public_link(item.findtext("link") or "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(SearchResult(title=title or urlsplit(url).hostname, url=url))
        if len(results) >= limit:
            break
    return results


def parse_duckduckgo_results(html: bytes, limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "lxml")
    if soup.select_one('#challenge-form, #anomaly-form, .anomaly-modal'):
        raise ScrapeError("DuckDuckGo search is temporarily unavailable. Please try again.", 503)
    results, seen_urls = [], set()
    for anchor in soup.select('a.result-link'):
        try:
            link = urljoin(DUCKDUCKGO_URL, anchor.get('href') or '')
            parsed = urlsplit(link)
            if parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"}:
                if parsed.path != '/l/':
                    continue
                link = parse_qs(parsed.query).get('uddg', [''])[0]
        except ValueError:
            continue
        url = _public_link(link)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(SearchResult(title=anchor.get_text(' ', strip=True) or urlsplit(url).hostname, url=url))
        if len(results) >= limit:
            break
    if not results and not soup.select_one('.no-results, .no-results__title'):
        raise ScrapeError("DuckDuckGo search returned no usable result links. Please try again.", 503)
    return results


def search_web(keyword: str, limit: int = 15, *, provider: str = "DuckDuckGo") -> tuple[list[SearchResult], str]:
    """Only discover URLs here: search snippets are never model inputs."""
    if provider not in SEARCH_PROVIDERS:
        raise ValueError("Unsupported search provider")
    phrase = " ".join(keyword.replace('"', ' ').split())
    # RSS is a fallback only: it can return unrelated URLs for a correct query.
    # Do not overconstrain HTML search with quotes; all extracted pages still
    # have to pass the same complete-topic relevance check.
    query = f'"{phrase}"' if provider == "Bing" and len(phrase.split()) > 1 else phrase
    url = DUCKDUCKGO_URL if provider == "DuckDuckGo" else SEARCH_URL
    params = {"q": query} if provider == "DuckDuckGo" else {"q": query, "format": "rss", "setlang": "en-US"}
    user_agent = SEARCH_USER_AGENT if provider == "DuckDuckGo" else USER_AGENT
    try:
        with httpx.Client(timeout=15, follow_redirects=False, headers={"User-Agent": user_agent}) as client:
            with client.stream("GET", url, params=params) as response:
                if response.status_code != 200:
                    raise ScrapeError("Web search is temporarily unavailable. Please try again.", 503)
                chunks, size = [], 0
                for chunk in response.iter_bytes(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ScrapeError("Web search returned an oversized response. Please try again.", 503)
                    chunks.append(chunk)
                content = b"".join(chunks)
                results = (parse_duckduckgo_results(content, limit) if provider == "DuckDuckGo" else
                           parse_results(content, limit))
    except httpx.HTTPError as exc:
        raise ScrapeError("Web search is temporarily unavailable. Please try again.", 503) from exc
    if not results:
        raise ScrapeError("No web results found. Try a more specific keyword or phrase.", 404)
    return results, provider
