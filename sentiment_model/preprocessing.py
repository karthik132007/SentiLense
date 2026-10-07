"""Stable, importable preprocessing shared by training and saved pipelines."""

import html
import re
import unicodedata

import ftfy

URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION = re.compile(r"(?<!\w)@[\w]+", re.UNICODE)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd\ud800-\udfff]")
ASCII_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE = re.compile(r"\s+")
TOKENS = re.compile(r"<3|[<>]?[:;=8][\-^']?[)(/\\dpo]+|\w+(?:'\w+)*|[^\w\s]", re.UNICODE)


def _replace_url(match):
    value = match.group()
    trailing_punctuation = value[len(value.rstrip(".,!?;:)]}")):]
    return " urltoken " + trailing_punctuation


def normalize_text(text: str) -> str:
    """Repair Unicode, unescape HTML, and replace URL/mention identity.

    Retain negation, hashtags, emoticons, emojis, and punctuation. No stemming,
    stopword removal, or ASCII-only conversion is performed.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # Replace before ftfy can delete controls and accidentally join words.
    text = ASCII_CONTROL.sub(" ", text)
    if not text.isascii():
        text = ftfy.fix_text(text, uncurl_quotes=False)
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = unicodedata.normalize("NFC", text)
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'"}))
    text = CONTROL.sub(" ", text)
    text = URL.sub(_replace_url, text)
    text = MENTION.sub(" usertoken ", text)
    return WHITESPACE.sub(" ", text.lower()).strip()


def normalize_batch(texts):
    return [normalize_text(text) for text in texts]


def tokenize(text: str) -> list[str]:
    return TOKENS.findall(text)
