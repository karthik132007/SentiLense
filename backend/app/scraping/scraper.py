"""Web scraping engine: fetch + extract human-readable article content."""

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10
MAX_BYTES = 2 * 1024 * 1024  # 2 MB response cap
USER_AGENT = "SentiLenseBot/1.0 (+academic sentiment analysis project)"

BLOCKED_HOSTNAMES = {"localhost"}


class ScrapeError(Exception):
    """Raised for any fetch/extract failure with a user-safe message."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ScrapedPage:
    url: str
    title: str
    domain: str
    description: str
    text: str


def validate_url(raw_url: str) -> str:
    """Validate scheme + reject obvious SSRF targets. Returns normalised URL."""
    raw_url = (raw_url or "").strip()
    if not raw_url:
        raise ScrapeError("URL must not be empty.", 422)
    if len(raw_url) > 2048:
        raise ScrapeError("URL is too long.", 422)
    try:
        parsed = urlparse(raw_url)
    except Exception:
        raise ScrapeError("Invalid URL.", 422)
    if parsed.scheme not in ("http", "https"):
        raise ScrapeError("Only http and https URLs are supported.", 422)
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ScrapeError("Invalid URL: missing host.", 422)
    if parsed.username is not None or parsed.password is not None:
        raise ScrapeError("URLs containing credentials are not supported.", 422)
    try:
        parsed.port
    except ValueError as exc:
        raise ScrapeError("Invalid URL port.", 422) from exc
    if host in BLOCKED_HOSTNAMES:
        raise ScrapeError("Access to that host is not allowed.", 400)
    # Block literal IP hosts in private/loopback ranges
    try:
        ip = ipaddress.ip_address(host)
        if _is_blocked_ip(ip):
            raise ScrapeError("Access to that host is not allowed.", 400)
    except ValueError:
        pass  # hostname, will resolve below
    _check_dns(host)
    return raw_url


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _check_dns(host: str) -> None:
    """Resolve hostname and block private/internal IPs (SSRF protection)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ScrapeError("Could not resolve host (unreachable website).", 400)
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            logger.warning("Blocked SSRF attempt to %s -> %s", host, ip_str)
            raise ScrapeError("Access to that host is not allowed.", 400)


def fetch_html(url: str) -> tuple[str, str]:
    """Download HTML with timeout + size + content-type checks.

    Returns (html, final_url). Raises ScrapeError on failure.
    """
    try:
        with httpx.Client(
            timeout=TIMEOUT_SECONDS,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
        ) as client:
            for redirect_count in range(6):
                # Validate EVERY destination before requesting it, including
                # redirects from public websites to internal addresses.
                url = validate_url(url)
                with client.stream("GET", url) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise ScrapeError("Website returned an invalid redirect.", 400)
                        if redirect_count == 5:
                            raise ScrapeError("Website returned too many redirects.", 400)
                        url = str(resp.url.join(location))
                        continue
                    if resp.status_code >= 400:
                        raise ScrapeError(f"Website returned HTTP {resp.status_code}.", 400)
                    ctype = resp.headers.get("content-type", "")
                    if ctype and "html" not in ctype.lower() and "text" not in ctype.lower():
                        raise ScrapeError("Unsupported content type. Only web pages can be analyzed.", 400)
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > MAX_BYTES:
                            raise ScrapeError("Page is too large to analyze.", 400)
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    final_url = str(resp.url)
                    encoding = resp.encoding or "utf-8"
                    break
    except ScrapeError:
        raise
    except httpx.TimeoutException:
        raise ScrapeError("Request timed out fetching the URL.", 400)
    except httpx.ConnectError:
        raise ScrapeError("Could not connect to the website (unreachable).", 400)
    except httpx.HTTPError as e:
        raise ScrapeError(f"Failed to fetch URL: {e}.", 400)
    try:
        return raw.decode(encoding, errors="replace"), final_url
    except Exception:
        return raw.decode("utf-8", errors="replace"), final_url


def extract_content(html: str, url: str) -> ScrapedPage:
    """Extract title/domain/description/main text from raw HTML."""
    parsed = urlparse(url)
    domain = parsed.hostname or ""

    # Try trafilatura first (better boilerplate removal)
    text_via_trafi: str | None = None
    try:
        from trafilatura import extract as trafi_extract

        text_via_trafi = trafi_extract(
            html, include_comments=False, include_tables=False
        )
    except Exception as e:
        logger.debug("trafilatura extraction failed: %s", e)

    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()

    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()
    if not description:
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            description = og_desc["content"].strip()

    # Remove boilerplate tags
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "button", "select", "input", "iframe"]):
        tag.decompose()
    # Remove common boilerplate by class/id
    for el in soup.find_all(attrs={"class": re.compile(
            r"nav|menu|sidebar|footer|header|cookie|popup|modal|advert|banner",
            re.I)}):
        if el.parent is not None:
            el.decompose()

    main = soup.find("main") or soup.find("article")
    main_text = _clean_text(main.get_text(separator="\n")) if main else ""
    body = soup.body or soup
    bs_text = _clean_text(body.get_text(separator="\n"))
    trafi_text = _clean_text(text_via_trafi) if text_via_trafi else ""

    # Article extraction removes related stories and service links. Choosing
    # the longest candidate favored the entire site chrome on news pages.
    if main_text and len(main_text) >= 50:
        final_text = main_text
    elif trafi_text and len(trafi_text) >= 50:
        final_text = trafi_text
    else:
        final_text = main_text or trafi_text or bs_text

    if not final_text or len(final_text) < 20:
        raise ScrapeError("Could not extract meaningful content from the page.", 400)

    return ScrapedPage(
        url=url, title=title or domain, domain=domain,
        description=description, text=final_text,
    )


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) > 1]
    # drop very short nav-ish lines is handled by boilerplate removal;
    # collapse whitespace
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    return joined.strip()


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_units(text: str, max_chars: int = 500, max_units: int = 100) -> list[str]:
    """Split long content into sentences / reasonable chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = SENTENCE_SPLIT.split(text)
    units: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                units.append(buf)
            if len(s) > max_chars:
                # hard-split very long sentence
                for i in range(0, len(s), max_chars):
                    units.append(s[i:i + max_chars].strip())
                buf = ""
            else:
                buf = s
        if len(units) >= max_units:
            break
    if buf and len(units) < max_units:
        units.append(buf)
    return [u for u in units if len(u) >= 3][:max_units]


def scrape_url(raw_url: str) -> ScrapedPage:
    url = validate_url(raw_url)
    html, final_url = fetch_html(url)
    return extract_content(html, final_url)
