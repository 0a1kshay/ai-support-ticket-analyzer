"""
Deterministic Query Engine.

Receives a validated StructuredQuery and executes it against the Pandas
DataFrame. The LLM never touches this code path.

Supported operations:
  COUNT_TICKETS      – count rows matching filters
  AVERAGE_METRIC     – average / min / max of a numeric field
  LIST_TICKETS       – return individual rows matching filters
  GROUP_BY           – group by a field, compute aggregate
  TOP_AGENT_RESOLVED – agent who resolved the most tickets
  TOP_AGENT_RATING   – agent with highest/lowest avg customer rating
  TOP_CATEGORY       – category with most/fewest tickets
  AGENT_RANKING      – rank all agents by resolved count or rating
  CATEGORY_RANKING   – rank all categories by ticket count
  ANOMALY_QUERY      – delegate to anomaly detector
  UNSUPPORTED        – return a helpful error
"""
from __future__ import annotations
import logging
from typing import Any, Optional

import pandas as pd

from app.models.schemas import (
    AggregationOperation, GroupByField, Intent,
    MetricField, StructuredQuery, TicketFilters
)
from app.utils.helpers import parse_date_filter, round_2

logger = logging.getLogger(__name__)

# Dataset reference date for temporal context (end of dataset range)
DATASET_REFERENCE_DATE_STR = "2024-03-30"


class QueryEngineError(Exception):
    pass


class QueryEngine:
    """
    Stateless engine; receives the DataFrame each call so it always
    reflects the current in-memory data.
    """

    def execute(self, query: StructuredQuery, df: pd.DataFrame) -> dict[str, Any]:
        """
        Main entry point. Returns a dict with at least 'answer' and 'data' keys.
        """
        intent = query.intent

        # Apply date filter first if present (before other filters)
        df = self._apply_date_filter(df, query)

        if intent == Intent.COUNT_TICKETS:
            return self._count_tickets(query, df)
        elif intent == Intent.AVERAGE_METRIC:
            return self._average_metric(query, df)
        elif intent == Intent.LIST_TICKETS:
            return self._list_tickets(query, df)
        elif intent == Intent.GROUP_BY:
            return self._group_by(query, df)
        elif intent == Intent.TOP_AGENT_RESOLVED:
            return self._top_agent_resolved(query, df)
        elif intent == Intent.TOP_AGENT_RATING:
            return self._top_agent_rating(query, df)
        elif intent == Intent.TOP_CATEGORY:
            return self._top_category(query, df)
        elif intent == Intent.AGENT_RANKING:
            return self._agent_ranking(query, df)
        elif intent == Intent.CATEGORY_RANKING:
            return self._category_ranking(query, df)
        elif intent == Intent.ANOMALY_QUERY:
            return {
                "answer": "Routing to the anomaly detection service. Please use the /api/anomalies endpoint for detailed anomaly information.",
                "data": {"route": "anomalies"},
                "route_to_anomalies": True,
            }
        elif intent == Intent.UNSUPPORTED:
            reason = query.unsupported_reason or "This question cannot be mapped to a supported analytics operation."
            return {
                "answer": f"I couldn't answer that question. {reason}\n\nSupported operations: count tickets, average/min/max metrics, list tickets with filters, group by category/priority/status/agent, agent and category rankings, anomaly detection.",
                "data": {"supported_capabilities": [
                    "Count tickets (with filters)",
                    "Average / min / max of response_time_hrs, resolution_time_hrs, customer_rating",
                    "List tickets matching filters",
                    "Group tickets by category, priority, status, or agent",
                    "Top agent by resolved count or customer rating",
                    "Category with most tickets",
                    "Agent and category rankings",
                    "Anomaly detection",
                ]},
            }
        else:
            raise QueryEngineError(f"Unhandled intent: {intent}")

    # ------------------------------------------------------------------
    # Filter application
    # ------------------------------------------------------------------

    def _apply_filters(self, df: pd.DataFrame,
                       filters: Optional[TicketFilters]) -> pd.DataFrame:
        if filters is None:
            return df

        if filters.status:
            if filters.status.lower() == "unresolved":
                df = df[df["status"].isin(["Open", "Escalated"])]
            else:
                df = df[df["status"] == filters.status]
        if filters.priority:
            df = df[df["priority"] == filters.priority]
        if filters.category:
            df = df[df["category"] == filters.category]
        if filters.agent_id:
            df = df[df["agent_id"] == filters.agent_id]
        if filters.min_rating is not None:
            df = df[df["customer_rating"] >= filters.min_rating]
        if filters.max_rating is not None:
            df = df[df["customer_rating"] <= filters.max_rating]
        if filters.min_resolution_hours is not None:
            df = df[df["resolution_time_hrs"] >= filters.min_resolution_hours]
        if filters.max_resolution_hours is not None:
            df = df[df["resolution_time_hrs"] <= filters.max_resolution_hours]
        if filters.not_resolved_within_hours is not None:
            thresh = filters.not_resolved_within_hours
            df = df[
                (df["status"].isin(["Open", "Escalated"])) |
                (df["resolution_time_hrs"] > thresh)
            ]
        if filters.min_response_hours is not None:
            df = df[df["response_time_hrs"] >= filters.min_response_hours]
        if filters.max_response_hours is not None:
            df = df[df["response_time_hrs"] <= filters.max_response_hours]
        if filters.older_than_hours is not None:
            ref = pd.Timestamp(DATASET_REFERENCE_DATE_STR)
            age_hours = (ref - df["created_at"]).dt.total_seconds() / 3600
            df = df[age_hours >= filters.older_than_hours]
        return df

    def _apply_date_filter(self, df: pd.DataFrame, query: StructuredQuery) -> pd.DataFrame:
        if query.date_filter is None:
            return df
        from datetime import date
        ref = date.fromisoformat(DATASET_REFERENCE_DATE_STR)
        try:
            start, end = parse_date_filter(
                query.date_filter.filter_type,
                query.date_filter.date_from,
                query.date_filter.date_to,
                reference_date=ref,
            )
            df = df[(df["created_at"] >= start) & (df["created_at"] <= end)]
        except Exception as exc:
            logger.warning("Date filter error: %s", exc)
        return df

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _count_tickets(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        filtered = self._apply_filters(df, query.filters)
        count = len(filtered)
        desc = self._describe_filters(query.filters)
        answer = f"There {'is' if count == 1 else 'are'} {count} ticket{'s' if count != 1 else ''}{desc}."
        return {"answer": answer, "data": {"count": count}}

    def _average_metric(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        filtered = self._apply_filters(df, query.filters)
        metric = query.metric
        if metric is None:
            return {"answer": "No metric specified for averaging.", "data": {}}

        col = metric.value if hasattr(metric, 'value') else str(metric)
        series = filtered[col].dropna()

        if series.empty:
            return {"answer": f"No data available for {col} with the given filters.", "data": {}}

        agg = query.aggregation or AggregationOperation.AVERAGE
        if agg == AggregationOperation.AVERAGE or agg == "average":
            result = round_2(series.mean())
            op_label = "average"
        elif agg == AggregationOperation.MIN or agg == "min":
            result = round_2(series.min())
            op_label = "minimum"
        elif agg == AggregationOperation.MAX or agg == "max":
            result = round_2(series.max())
            op_label = "maximum"
        else:
            result = round_2(series.mean())
            op_label = "average"

        metric_labels = {
            "response_time_hrs": "response time (hours)",
            "resolution_time_hrs": "resolution time (hours)",
            "customer_rating": "customer rating",
        }
        label = metric_labels.get(col, col)
        desc = self._describe_filters(query.filters)
        answer = f"The {op_label} {label}{desc} is {result}."

        return {
            "answer": answer,
            "data": {op_label: result, "metric": col, "sample_size": int(series.count())},
        }

    def _list_tickets(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        filtered = self._apply_filters(df, query.filters)

        if query.sort:
            field = query.sort.field
            if field in filtered.columns:
                ascending = query.sort.direction == "asc"
                filtered = filtered.sort_values(field, ascending=ascending)

        limit = query.limit or 50
        filtered = filtered.head(limit)

        if filtered.empty:
            desc = self._describe_filters(query.filters)
            return {
                "answer": f"No tickets found{desc}.",
                "data": {"rows": [], "total": 0},
            }

        cols = ["ticket_id", "created_at", "category", "priority", "status",
                "response_time_hrs", "resolution_time_hrs", "agent_id",
                "customer_rating", "issue_summary"]
        rows = (
            filtered[cols]
            .copy()
            .assign(created_at=filtered["created_at"].dt.strftime("%Y-%m-%d %H:%M"))
            .to_dict(orient="records")
        )

        desc = self._describe_filters(query.filters)
        answer = f"Found {len(rows)} ticket{'s' if len(rows) != 1 else ''}{desc}."
        return {"answer": answer, "data": {"rows": rows, "total": len(rows)}}

    def _group_by(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        filtered = self._apply_filters(df, query.filters)
        group_field = query.group_by
        if group_field is None:
            return {"answer": "No group-by field specified.", "data": {}}

        col = group_field.value if hasattr(group_field, 'value') else str(group_field)

        metric = query.metric
        agg = query.aggregation or AggregationOperation.COUNT

        if agg in (AggregationOperation.COUNT, "count") or metric is None:
            grouped = (
                filtered.groupby(col)
                .size()
                .reset_index(name="count")
            )
        else:
            metric_col = metric.value if hasattr(metric, 'value') else str(metric)
            agg_str = agg.value if hasattr(agg, 'value') else str(agg)
            agg_map = {
                "average": "mean", "min": "min", "max": "max",
                "sum": "sum", "count": "count",
            }
            pandas_agg = agg_map.get(agg_str, "mean")
            grouped = (
                filtered.groupby(col)[metric_col]
                .agg(pandas_agg)
                .reset_index()
                .rename(columns={metric_col: f"{agg_str}_{metric_col}"})
            )
            grouped[f"{agg_str}_{metric_col}"] = grouped[f"{agg_str}_{metric_col}"].round(2)

        if query.sort:
            sort_col = grouped.columns[-1]
            ascending = query.sort.direction == "asc"
            grouped = grouped.sort_values(sort_col, ascending=ascending)

        rows = grouped.to_dict(orient="records")
        answer = f"Ticket distribution by {col}:\n" + "\n".join(
            f"  {row[col]}: {list(row.values())[1]}" for row in rows
        )
        return {"answer": answer, "data": {"rows": rows, "group_by": col}}

    def _top_agent_resolved(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        resolved = df[df["status"] == "Resolved"]
        if resolved.empty:
            return {"answer": "No resolved tickets found.", "data": {}}

        ranking = (
            resolved.groupby("agent_id")
            .size()
            .reset_index(name="resolved_count")
            .sort_values("resolved_count", ascending=False)
        )

        sort_dir = "desc"
        if query.sort:
            sort_dir = query.sort.direction

        if sort_dir == "asc":
            ranking = ranking.sort_values("resolved_count", ascending=True)
        else:
            ranking = ranking.sort_values("resolved_count", ascending=False)

        top = ranking.iloc[0]
        agent = top["agent_id"]
        count = int(top["resolved_count"])

        answer = (
            f"Agent {agent} resolved the {'most' if sort_dir == 'desc' or not query.sort else 'fewest'} tickets "
            f"with {count} resolved ticket{'s' if count != 1 else ''}."
        )
        return {
            "answer": answer,
            "data": {
                "agent_id": agent,
                "resolved_count": count,
                "full_ranking": ranking.to_dict(orient="records"),
            },
        }

    def _top_agent_rating(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        rated = df[df["customer_rating"].notna()]
        if rated.empty:
            return {"answer": "No rated tickets found.", "data": {}}

        ranking = (
            rated.groupby("agent_id")["customer_rating"]
            .mean()
            .reset_index()
            .rename(columns={"customer_rating": "avg_rating"})
        )
        ranking["avg_rating"] = ranking["avg_rating"].round(2)

        ascending = True  # default: lowest rating first
        if query.sort and query.sort.direction == "desc":
            ascending = False

        ranking = ranking.sort_values("avg_rating", ascending=ascending)
        top = ranking.iloc[0]
        agent = top["agent_id"]
        rating = round_2(top["avg_rating"])

        qualifier = "lowest" if ascending else "highest"
        answer = (
            f"Agent {agent} has the {qualifier} average customer rating of {rating}/5.0."
        )
        return {
            "answer": answer,
            "data": {
                "agent_id": agent,
                "avg_rating": rating,
                "full_ranking": ranking.to_dict(orient="records"),
            },
        }

    def _top_category(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        filtered = self._apply_filters(df, query.filters)
        ranking = (
            filtered.groupby("category")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        if ranking.empty:
            return {"answer": "No tickets found.", "data": {}}

        ascending = False
        if query.sort and query.sort.direction == "asc":
            ascending = True
            ranking = ranking.sort_values("count", ascending=True)

        top = ranking.iloc[0]
        cat = top["category"]
        count = int(top["count"])

        qualifier = "fewest" if ascending else "most"
        answer = f'The "{cat}" category has the {qualifier} tickets with {count}.'
        return {
            "answer": answer,
            "data": {
                "category": cat,
                "count": count,
                "full_ranking": ranking.to_dict(orient="records"),
            },
        }

    def _agent_ranking(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        metric = query.metric
        if metric and str(metric) in ("customer_rating", MetricField.CUSTOMER_RATING):
            return self._top_agent_rating(query, df)
        else:
            return self._top_agent_resolved(query, df)

    def _category_ranking(self, query: StructuredQuery, df: pd.DataFrame) -> dict:
        return self._top_category(query, df)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _describe_filters(self, filters: Optional[TicketFilters]) -> str:
        if filters is None:
            return ""
        parts = []
        if filters.status:
            if filters.status.lower() == "unresolved":
                parts.append("that are unresolved (Open or Escalated)")
            else:
                parts.append(f"with status '{filters.status}'")
        if filters.priority:
            parts.append(f"with priority '{filters.priority}'")
        if filters.category:
            parts.append(f"in category '{filters.category}'")
        if filters.agent_id:
            parts.append(f"assigned to agent '{filters.agent_id}'")
        if filters.older_than_hours:
            parts.append(f"older than {filters.older_than_hours} hours")
        if filters.not_resolved_within_hours is not None:
            parts.append(f"not resolved within {filters.not_resolved_within_hours} hours")
        if filters.max_resolution_hours is not None:
            parts.append(f"resolved within {filters.max_resolution_hours} hours")
        return (" " + " and ".join(parts)) if parts else ""
