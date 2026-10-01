"""Natural-language query endpoint."""
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import ValidationError

from app.models.schemas import (
    QueryRequest, QueryResponse, StructuredQuery, Intent
)
from app.services.llm_service import LLMUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])


@router.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Natural-language ticket query",
    description=(
        "Accepts a natural-language question, uses the configured LLM to extract "
        "structured intent, validates it with Pydantic, then executes deterministically "
        "against the dataset. The LLM never performs calculations."
    ),
)
async def query_tickets(body: QueryRequest, request: Request) -> QueryResponse:
    question = body.question
    logger.info("Query received: %.200s", question)

    llm_provider = request.app.state.llm_provider
    intent_parser = request.app.state.intent_parser
    query_engine = request.app.state.query_engine
    formatter = request.app.state.response_formatter
    ds = request.app.state.data_service
    anomaly_detector = request.app.state.anomaly_detector

    # --- Step 1: Parse intent via LLM ---
    try:
        raw_query = intent_parser.parse(question)
    except LLMUnavailableError as exc:
        logger.warning("LLM unavailable: %s", exc)
        return formatter.format_error(
            question,
            "The AI query parser is currently unavailable. "
            "Please ensure GROQ_API_KEY is set in .env or start Ollama. "
            f"Details: {exc}",
            intent="LLM_UNAVAILABLE",
        )
    except ValueError as exc:
        logger.warning("LLM returned invalid JSON: %s", exc)
        return formatter.format_error(
            question,
            "The AI returned an unrecognised response format. Please try rephrasing your question.",
            intent="PARSE_ERROR",
        )

    # --- Step 2: Validate with Pydantic ---
    try:
        structured_query = StructuredQuery.model_validate(raw_query)
    except ValidationError as exc:
        logger.warning("StructuredQuery validation failed: %s", exc)
        return formatter.format_error(
            question,
            f"The query could not be validated: {exc.errors()[0]['msg']}. "
            "Please rephrase your question.",
            intent="VALIDATION_ERROR",
        )

    # --- Step 3: Execute deterministically ---
    df = ds.get_dataframe()

    try:
        if structured_query.intent == Intent.ANOMALY_QUERY:
            anomalies = anomaly_detector.detect_all(df)
            summary = anomaly_detector.get_summary(anomalies)
            top_anomalies = [a.model_dump() for a in anomalies[:10]]
            engine_result = {
                "answer": (
                    f"Found {summary['total_count']} anomalies in the dataset. "
                    f"By type: {summary['by_type']}. "
                    f"By severity: {summary['by_severity']}. "
                    "Visit /api/anomalies for full details."
                ),
                "data": {
                    "total_anomalies": summary["total_count"],
                    "by_type": summary["by_type"],
                    "by_severity": summary["by_severity"],
                    "sample_anomalies": top_anomalies,
                },
            }
        else:
            engine_result = query_engine.execute(structured_query, df)
    except Exception as exc:
        logger.error("Query engine error: %s", exc, exc_info=True)
        return formatter.format_error(
            question,
            "An internal error occurred while processing your query. Please try again.",
            intent=structured_query.intent.value,
        )

    return formatter.format(question, structured_query, engine_result)
