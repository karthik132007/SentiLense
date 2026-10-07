"""Schemas for the SentiLense backend API."""

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class TextAnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float
    scores: Dict[str, float]


class UrlAnalyzeRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


class SentenceResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]


class UrlAnalyzeResponse(BaseModel):
    url: str
    title: str
    domain: str
    overall_sentiment: str
    confidence: float
    scores: Dict[str, float]
    distribution: Dict[str, float]
    statistics: Dict[str, int]
    results: List[SentenceResult]
    analysis_model: str
    aggregation: str


class KeywordAnalyzeRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)
    max_sources: int = Field(default=5, ge=1, le=5, strict=True)

    @field_validator("keyword")
    @classmethod
    def normalize_keyword(cls, value):
        value = " ".join(value.split())
        if not value or any(ord(character) < 32 for character in value):
            raise ValueError("Enter a non-empty keyword or phrase.")
        return value


class SkippedSource(BaseModel):
    url: str
    title: str
    reason: str


class TopicEvidence(BaseModel):
    method: str
    title_matches: bool
    matched_passages: int
    extracted_passages: int
    extracted_words: int


class KeywordSource(UrlAnalyzeResponse):
    topic_evidence: TopicEvidence


class KeywordAnalyzeResponse(BaseModel):
    keyword: str
    search_provider: str
    searched_at_utc: str
    overall_sentiment: str
    confidence: float
    scores: Dict[str, float]
    distribution: Dict[str, float]
    statistics: Dict[str, int]
    aggregation: str
    sources: List[KeywordSource]
    skipped_sources: List[SkippedSource]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
