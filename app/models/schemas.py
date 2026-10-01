"""
Pydantic schemas for the AI Support Ticket Analyzer.
All request/response models and the structured query representation.
The LLM may only emit values from the allowlisted enumerations below.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Allowlisted enumerations – the LLM may only emit values from these sets
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    COUNT_TICKETS = "COUNT_TICKETS"
    AVERAGE_METRIC = "AVERAGE_METRIC"
    LIST_TICKETS = "LIST_TICKETS"
    GROUP_BY = "GROUP_BY"
    TOP_AGENT_RESOLVED = "TOP_AGENT_RESOLVED"
    TOP_AGENT_RATING = "TOP_AGENT_RATING"
    TOP_CATEGORY = "TOP_CATEGORY"
    AGENT_RANKING = "AGENT_RANKING"
    CATEGORY_RANKING = "CATEGORY_RANKING"
    ANOMALY_QUERY = "ANOMALY_QUERY"
    UNSUPPORTED = "UNSUPPORTED"


class AggregationOperation(str, Enum):
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    SUM = "sum"


class MetricField(str, Enum):
    RESPONSE_TIME = "response_time_hrs"
    RESOLUTION_TIME = "resolution_time_hrs"
    CUSTOMER_RATING = "customer_rating"


class GroupByField(str, Enum):
    CATEGORY = "category"
    PRIORITY = "priority"
    STATUS = "status"
    AGENT = "agent_id"


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class DateFilterType(str, Enum):
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    AFTER = "after"
    BEFORE = "before"
    BETWEEN = "between"


# ---------------------------------------------------------------------------
# Nested filter / sort models
# ---------------------------------------------------------------------------

class DateFilter(BaseModel):
    filter_type: DateFilterType
    date_from: Optional[str] = None   # ISO date for AFTER / BETWEEN start
    date_to: Optional[str] = None     # ISO date for BEFORE / BETWEEN end


class TicketFilters(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    category: Optional[str] = None
    agent_id: Optional[str] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    min_resolution_hours: Optional[float] = None
    max_resolution_hours: Optional[float] = None
    min_response_hours: Optional[float] = None
    max_response_hours: Optional[float] = None
    not_resolved_within_hours: Optional[float] = None
    older_than_hours: Optional[float] = None   # ticket age constraint

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v):
        if v is not None and isinstance(v, str):
            v_title = v.strip().title()
            if v_title == "Unresolved":
                return "Unresolved"
        allowed = {"Open", "Resolved", "Escalated", "Unresolved", None}
        if v not in allowed:
            raise ValueError(f"Invalid status '{v}'. Allowed: {sorted(allowed - {None})}")
        return v

    @field_validator("priority", mode="before")
    @classmethod
    def validate_priority(cls, v):
        allowed = {"Low", "Medium", "High", "Critical", None}
        if v not in allowed:
            raise ValueError(f"Invalid priority '{v}'. Allowed: {sorted(allowed - {None})}")
        return v

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v):
        allowed = {"General", "Billing", "Technical", None}
        if v not in allowed:
            raise ValueError(f"Invalid category '{v}'. Allowed: {sorted(allowed - {None})}")
        return v


class SortSpec(BaseModel):
    field: str
    direction: SortDirection = SortDirection.DESC


# ---------------------------------------------------------------------------
# Core structured query (LLM output → Pydantic validation → Query Engine)
# ---------------------------------------------------------------------------

class StructuredQuery(BaseModel):
    intent: Intent
    filters: Optional[TicketFilters] = None
    metric: Optional[MetricField] = None
    aggregation: Optional[AggregationOperation] = None
    group_by: Optional[GroupByField] = None
    sort: Optional[SortSpec] = None
    limit: Optional[int] = Field(default=None, ge=1, le=500)
    date_filter: Optional[DateFilter] = None
    unsupported_reason: Optional[str] = None  # set when intent is UNSUPPORTED


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          description="Natural-language question about support tickets.")

    @field_validator("question", mode="before")
    @classmethod
    def strip_and_validate(cls, v):
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        return v


class QueryResponse(BaseModel):
    question: str
    intent: str
    answer: str
    data: dict[str, Any] = {}
    structured_query: Optional[dict[str, Any]] = None


class AnomalyRecord(BaseModel):
    ticket_id: str
    anomaly_type: str
    severity: str
    reason: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


class AnomalyResponse(BaseModel):
    total_count: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    anomalies: list[AnomalyRecord]


class StatsResponse(BaseModel):
    total_tickets: int
    open_tickets: int
    resolved_tickets: int
    escalated_tickets: int
    critical_tickets: int
    avg_response_time_hrs: Optional[float]
    avg_resolution_time_hrs: Optional[float]
    avg_customer_rating: Optional[float]
    anomaly_count: int
    categories: dict[str, int]
    priorities: dict[str, int]
    top_agents_by_resolved: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    total_rows: int
    llm_provider: str
    llm_available: bool
    message: str
