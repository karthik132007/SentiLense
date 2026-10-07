"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import os

from app.api.routes import router
from app.ml.inference import get_adapter, get_review_adapter

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app):
    get_adapter()
    get_review_adapter()
    yield


app = FastAPI(title="SentiLense API", version="1.0.0", lifespan=lifespan)
ROOT = Path(__file__).resolve().parents[2]
FRONTEND = Path(os.environ.get("SENTILENSE_FRONTEND_DIR", ROOT / "frontend/dist/sentiment-intelligence/browser"))

# Development CORS for the Angular frontend agent (localhost) + general dev use
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # Never turn missing API routes or assets into a successful HTML
            # response. Only client-side navigation gets the Angular shell.
            if exc.status_code == 404 and not path.startswith("api/") and "." not in Path(path).name:
                return await super().get_response("index.html", scope)
            raise


if (FRONTEND / "index.html").is_file():
    app.mount("/", SPAStaticFiles(directory=FRONTEND, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "SentiLense API is running. Build frontend/ to serve the application here. See /docs for the API."}
