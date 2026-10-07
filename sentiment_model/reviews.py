"""Additional movie-review document model; preserve the original tweet baseline."""

import os
from pathlib import Path
import re

from .inference import predict_sentiment


def review_model_dir() -> Path:
    return Path(os.environ.get('SENTILENSE_REVIEW_MODEL_DIR') or
                Path(__file__).resolve().parents[1] / 'model/reviews').resolve()


def is_movie_review(title: str, text: str) -> bool:
    """Conservative, disclosed domain routing, independent of sentiment scores.

    A review headline plus cinema vocabulary routes to the IMDb model; this
    does not decide polarity. Every label comes from trained probabilities.
    """
    if not re.search(r'\breview\b', title, re.IGNORECASE):
        return False
    if re.search(r'\b(?:movie|film)\b', title, re.IGNORECASE):
        return True
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return len(words & {'film', 'movie', 'cinema', 'director', 'screenplay', 'actor', 'acting'}) >= 3


def predict_review(text: str) -> dict:
    return predict_sentiment(text, model_dir=review_model_dir())
