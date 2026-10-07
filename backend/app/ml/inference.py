"""Backend adapters for three-class fusion and binary review pipelines."""

from functools import lru_cache
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentiment_model import predict_sentiment
from sentiment_model.inference import _load
from sentiment_model.reviews import review_model_dir


class RealAdapter:
    IS_MOCK = False

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or os.environ.get("SENTILENSE_MODEL_DIR", REPO_ROOT / "model")).resolve()
        self.pipeline, self.metadata = _load(str(self.model_dir))

    def predict(self, text: str) -> dict:
        return predict_sentiment(text, model_dir=self.model_dir)

    def predict_many(self, texts: list[str]) -> list[dict]:
        if self.metadata.get("model_family") == "svm_roberta_fusion":
            return self.pipeline.predict_many(texts)
        return [self.predict(text) for text in texts]

    def classify_scores(self, scores: dict[str, float]) -> dict:
        if self.metadata.get("model_family") == "svm_roberta_fusion":
            from sentiment_model.fusion import score_result
            return score_result(scores)
        return self.classify_probability(scores["positive"])

    def classify_probability(self, positive_probability: float) -> dict:
        """Apply the trained neutral policy to a page's mean probability.

        Page averaging is descriptive aggregation; tweet validation does not
        establish page-level calibration or accuracy.
        """
        probability = float(positive_probability)
        if self.metadata.get("model_family") == "svm_roberta_fusion":
            raise ValueError("Three-class fusion requires all class scores; use classify_scores")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Expected a finite probability between zero and one")
        if probability >= self.metadata["positive_threshold"]:
            label = "positive"
        elif probability <= self.metadata["negative_threshold"]:
            label = "negative"
        else:
            label = "neutral"
        return {"sentiment": label, "confidence": max(probability, 1 - probability),
                "scores": {"positive": probability, "negative": 1 - probability}}


@lru_cache(maxsize=1)
def load_adapter():
    # Fail for missing/corrupt models rather than fabricating heuristic output.
    return RealAdapter()


def get_adapter():
    return load_adapter()


@lru_cache(maxsize=1)
def get_review_adapter():
    return RealAdapter(review_model_dir())


def model_status() -> dict:
    load_adapter()
    return {"model_loaded": True, "model_backend": "sentiment_model"}


def reset_adapter():
    load_adapter.cache_clear()
    get_review_adapter.cache_clear()
    _load.cache_clear()
