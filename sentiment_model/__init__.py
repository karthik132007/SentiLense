"""Backend-facing sentiment inference; model loading is lazy."""

from .inference import predict_sentiment

__all__ = ["predict_sentiment"]
