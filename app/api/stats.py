"""Statistics dashboard endpoint."""
from fastapi import APIRouter, Request
from app.models.schemas import StatsResponse

router = APIRouter(tags=["Statistics"])


@router.get(
    "/api/stats",
    response_model=StatsResponse,
    summary="Dashboard statistics",
    description="Returns aggregated KPIs and summary statistics for the dashboard.",
)
async def get_stats(request: Request) -> StatsResponse:
    ds = request.app.state.data_service
    anomaly_detector = request.app.state.anomaly_detector

    stats = ds.get_stats()
    df = ds.get_dataframe()
    anomalies = anomaly_detector.detect_all(df)

    return StatsResponse(
        total_tickets=stats["total_tickets"],
        open_tickets=stats["open_tickets"],
        resolved_tickets=stats["resolved_tickets"],
        escalated_tickets=stats["escalated_tickets"],
        critical_tickets=stats["critical_tickets"],
        avg_response_time_hrs=stats.get("avg_response_time_hrs"),
        avg_resolution_time_hrs=stats.get("avg_resolution_time_hrs"),
        avg_customer_rating=stats.get("avg_customer_rating"),
        anomaly_count=len(anomalies),
        categories=stats["categories"],
        priorities=stats["priorities"],
        top_agents_by_resolved=stats["top_agents_by_resolved"],
    )
