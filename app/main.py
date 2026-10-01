"""
AI Support Ticket Analyzer — FastAPI Application Entry Point.

Architecture:
    User → FastAPI → LLM Intent Parser → Pydantic Validation
    → Deterministic Query Engine → Pandas Dataset → Response

The LLM is used ONLY for natural-language understanding (intent extraction).
All calculations are performed deterministically in Python.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables from .env before importing services
load_dotenv()

from app.api import health, stats, anomalies, query
from app.services.data_service import DataService, DataServiceError
from app.services.llm_service import get_llm_provider, IntentParser
from app.services.query_engine import QueryEngine
from app.services.anomaly_detector import AnomalyDetector
from app.services.response_formatter import ResponseFormatter

from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: initialise all services
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("AI Support Ticket Analyzer starting up...")

    # Locate CSV
    csv_candidates = [
        os.getenv("CSV_PATH", ""),
        "data/support_tickets.csv",
        "support_tickets.csv",
    ]
    csv_path = next((p for p in csv_candidates if p and os.path.isfile(p)),
                    "data/support_tickets.csv")

    # Data service
    try:
        app.state.data_service = DataService(csv_path)
        logger.info("Dataset loaded: %d rows", app.state.data_service.row_count)
    except DataServiceError as exc:
        logger.error("FATAL: Could not load dataset: %s", exc)
        raise

    # LLM provider
    app.state.llm_provider = get_llm_provider()
    logger.info("LLM provider: %s (available=%s)",
                app.state.llm_provider.name,
                app.state.llm_provider.is_available())

    # Services
    app.state.intent_parser = IntentParser(app.state.llm_provider)
    app.state.query_engine = QueryEngine()
    app.state.anomaly_detector = AnomalyDetector()
    app.state.response_formatter = ResponseFormatter()

    logger.info("All services initialised successfully.")
    logger.info("=" * 60)
    yield

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Support Ticket Analyzer",
        description=(
            "An AI-powered customer support ticket analytics system. "
            "Natural-language questions are parsed by an LLM into structured intents, "
            "validated by Pydantic, then executed deterministically against the dataset. "
            "No LLM is involved in the actual calculations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS (allow all origins for local development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Include routers
    # ---------------------------------------------------------------------------
    app.include_router(health.router)
    app.include_router(stats.router)
    app.include_router(anomalies.router)
    app.include_router(query.router)

    # ---------------------------------------------------------------------------
    # Static files (UI)
    # ---------------------------------------------------------------------------
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_ui():
            return FileResponse(str(static_dir / "index.html"))
    else:
        @app.get("/", include_in_schema=False)
        async def root():
            return {"message": "AI Support Ticket Analyzer API", "docs": "/docs"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )
