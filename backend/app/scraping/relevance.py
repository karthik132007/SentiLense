"""Conservative lexical topic evidence, checked against extracted page content.

This is not semantic relevance or entity disambiguation. Requiring the complete
phrase avoids silently turning a multiword topic into its first word.
"""

import re
import unicodedata

from app.scraping.scraper import ScrapedPage, ScrapeError, split_units


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())
    normalized = []
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        if word == "centre":
            word = "center"
        normalized.append(word)
    return normalized


def mentions_topic(text: str, keyword: str) -> bool:
    topic, words = _tokens(keyword), _tokens(text)
    return bool(topic) and any(words[i:i + len(topic)] == topic for i in range(len(words) - len(topic) + 1))


def topic_passages(page: ScrapedPage, keyword: str) -> tuple[list[str], dict]:
    # Inspect the whole extracted document before imposing the inference cap.
    # Match individual sentences, so unrelated nearby tone is not averaged in.
    passages = [sentence.strip() for paragraph in page.text.splitlines()
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]
    matches = [passage for passage in passages if mentions_topic(passage, keyword)]
    title_matches = mentions_topic(page.title, keyword)
    if not matches or (not title_matches and len(matches) < 2 and len(matches) / len(passages) < 0.2):
        raise ScrapeError("Off-topic source: insufficient extracted content matching the complete topic.", 422)
    units = []
    for passage in matches:
        # Very long passages still need bounded model inputs; retain only chunks
        # containing the topic, rather than using the remainder of the article.
        units.extend(unit for unit in split_units(passage, max_units=max(100, len(passage) // 500 + 2))
                     if mentions_topic(unit, keyword))
        if len(units) >= 100:
            break
    if not units:
        raise ScrapeError("No analyzable passages matching the complete topic.", 422)
    return units[:100], {
        "method": "complete_phrase_with_plural_and_center_spelling_variants",
        "title_matches": title_matches,
        "matched_passages": len(matches),
        "extracted_passages": len(passages),
        "extracted_words": len(page.text.split()),
    }
