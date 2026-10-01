"""Health check endpoint."""
from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description="Returns application status, dataset info, and LLM availability.",
)
async def health_check(request: Request) -> HealthResponse:
    ds = request.app.state.data_service
    llm = request.app.state.llm_provider

    llm_available = False
    try:
        llm_available = llm.is_available()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if ds.is_loaded else "degraded",
        dataset_loaded=ds.is_loaded,
        total_rows=ds.row_count,
        llm_provider=llm.name,
        llm_available=llm_available,
        message=(
            "All systems operational."
            if ds.is_loaded and llm_available
            else (
                "Dataset loaded; LLM unavailable — configure GROQ_API_KEY or start Ollama."
                if ds.is_loaded
                else "Dataset not loaded."
            )
        ),
    )
