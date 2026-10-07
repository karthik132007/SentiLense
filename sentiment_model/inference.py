"""Load the saved pipeline once and apply validation-selected abstention."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import joblib

from .preprocessing import normalize_text


@lru_cache(maxsize=4)
def _load(model_dir: str):
    directory = Path(model_dir)
    metadata = json.loads((directory / "model_metadata.json").read_text())
    if metadata.get("model_family") == "svm_roberta_fusion":
        from .fusion import FusionModel
        return FusionModel(directory, metadata), metadata
    pipeline_path = directory / "sentiment_pipeline.joblib"
    checksum = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
    if checksum != metadata["pipeline_sha256"]:
        raise ValueError("Pipeline and metadata checksum mismatch; deploy both together")
    negative = metadata["negative_threshold"]
    positive = metadata["positive_threshold"]
    if not 0 <= negative < 0.5 < positive <= 1:
        raise ValueError("Invalid uncertainty thresholds in model metadata")
    pipeline = joblib.load(pipeline_path)
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError("Expected binary model classes [0, 1]")
    return pipeline, metadata


def predict_sentiment(text: str, *, model_dir: str | Path | None = None) -> dict:
    """Return class scores from the saved model and the selected label's score.

    The fusion has a learned neutral class. Legacy binary experts retain their
    metadata-selected uncertainty band and max(binary scores) confidence.
    Set SENTILENSE_MODEL_DIR or pass model_dir to relocate model artifacts.
    """
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("text must contain non-whitespace content")
    directory = Path(model_dir or os.environ.get("SENTILENSE_MODEL_DIR") or
                     Path(__file__).resolve().parents[1] / "model").resolve()
    pipeline, metadata = _load(str(directory))
    if metadata.get("model_family") == "svm_roberta_fusion":
        return pipeline.predict_many([text])[0]
    # The pipeline receives RAW text: saved preprocessing always runs here.
    vector = pipeline[:-1].transform([text])
    if vector.nnz == 0:
        # Unknown vocabulary supplies no evidence, regardless of intercept.
        p_positive = 0.5
    else:
        p_positive = float(pipeline[-1].predict_proba(vector)[0, 1])
    if p_positive >= metadata["positive_threshold"]:
        sentiment = "positive"
    elif p_positive <= metadata["negative_threshold"]:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {
        "sentiment": sentiment,
        "confidence": max(p_positive, 1.0 - p_positive),
        "scores": {"positive": p_positive, "negative": 1.0 - p_positive},
    }
