"""API routes."""

import logging

from fastapi import APIRouter, HTTPException

from app.ml.inference import model_status
from app.schemas.api import (
    HealthResponse,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
    UrlAnalyzeRequest,
    UrlAnalyzeResponse,
    KeywordAnalyzeRequest,
    KeywordAnalyzeResponse,
)
from app.scraping.scraper import ScrapeError
from app.services.analysis import analyze_text, analyze_url
from app.services.keyword_analysis import analyze_keyword

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze/keyword", response_model=KeywordAnalyzeResponse)
def analyze_keyword_endpoint(body: KeywordAnalyzeRequest):
    try:
        return analyze_keyword(body.keyword, body.max_sources)
    except ScrapeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Keyword analysis failed")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc


@router.get("/health", response_model=HealthResponse)
def health():
    status = model_status()
    return {"status": "ok", **status}


@router.post("/analyze/text", response_model=TextAnalyzeResponse)
def analyze_text_endpoint(body: TextAnalyzeRequest):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    try:
        return analyze_text(text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("text analysis failed")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from e


@router.post("/analyze/url", response_model=UrlAnalyzeResponse)
def analyze_url_endpoint(body: UrlAnalyzeRequest):
    try:
        return analyze_url(body.url)
    except ScrapeError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.exception("url analysis failed")
        raise HTTPException(status_code=500, detail="Internal analysis error.")
