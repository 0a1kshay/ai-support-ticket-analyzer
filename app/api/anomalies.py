"""Anomaly detection endpoints."""
from typing import Optional
from fastapi import APIRouter, Query, Request
from app.models.schemas import AnomalyResponse

router = APIRouter(tags=["Anomalies"])


@router.get(
    "/api/anomalies",
    response_model=AnomalyResponse,
    summary="List detected anomalies",
    description=(
        "Returns all anomalies detected in the dataset using four deterministic rules:\n"
        "- LONG_RESOLUTION_TIME: resolution time > mean + 2*std\n"
        "- HIGH_PRIORITY_OVERDUE: High/Critical unresolved ticket older than 24h\n"
        "- SLOW_FIRST_RESPONSE: response time > mean + 2*std\n"
        "- LOW_CUSTOMER_RATING: resolved ticket rated <= 2/5"
    ),
)
async def get_anomalies(
    request: Request,
    severity: Optional[str] = Query(None, description="Filter by severity (CRITICAL, HIGH, MEDIUM)"),
    anomaly_type: Optional[str] = Query(None, description="Filter by anomaly type"),
    priority: Optional[str] = Query(None, description="Filter by ticket priority"),
    category: Optional[str] = Query(None, description="Filter by ticket category"),
) -> AnomalyResponse:
    ds = request.app.state.data_service
    detector = request.app.state.anomaly_detector

    df = ds.get_dataframe()

    # Optionally filter the dataframe before anomaly detection for priority/category
    if priority:
        df = df[df["priority"] == priority]
    if category:
        df = df[df["category"] == category]

    anomalies = detector.detect_all(df)

    # Post-filter by severity / type
    if severity:
        anomalies = [a for a in anomalies if a.severity == severity.upper()]
    if anomaly_type:
        anomalies = [a for a in anomalies if a.anomaly_type == anomaly_type.upper()]

    summary = detector.get_summary(anomalies)

    return AnomalyResponse(
        total_count=summary["total_count"],
        by_type=summary["by_type"],
        by_severity=summary["by_severity"],
        anomalies=anomalies,
    )
