"""Tests for the deterministic QueryEngine."""
import os
import pytest
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "support_tickets.csv")


@pytest.fixture(scope="module")
def df():
    from app.services.data_service import DataService
    svc = DataService(CSV_PATH)
    return svc.get_dataframe()


@pytest.fixture
def engine():
    from app.services.query_engine import QueryEngine
    return QueryEngine()


def make_query(**kwargs):
    from app.models.schemas import StructuredQuery, Intent
    defaults = {"intent": Intent.COUNT_TICKETS}
    defaults.update(kwargs)
    return StructuredQuery(**defaults)


class TestCountTickets:
    def test_count_all(self, engine, df):
        q = make_query(intent="COUNT_TICKETS")
        result = engine.execute(q, df)
        assert result["data"]["count"] == 500

    def test_count_open(self, engine, df):
        from app.models.schemas import TicketFilters
        q = make_query(intent="COUNT_TICKETS", filters=TicketFilters(status="Open"))
        result = engine.execute(q, df)
        expected = int((df["status"] == "Open").sum())
        assert result["data"]["count"] == expected

    def test_count_critical_open(self, engine, df):
        from app.models.schemas import TicketFilters
        q = make_query(
            intent="COUNT_TICKETS",
            filters=TicketFilters(status="Open", priority="Critical"),
        )
        result = engine.execute(q, df)
        expected = int(((df["status"] == "Open") & (df["priority"] == "Critical")).sum())
        assert result["data"]["count"] == expected

    def test_count_billing_open(self, engine, df):
        from app.models.schemas import TicketFilters
        q = make_query(
            intent="COUNT_TICKETS",
            filters=TicketFilters(status="Open", category="Billing"),
        )
        result = engine.execute(q, df)
        expected = int(((df["status"] == "Open") & (df["category"] == "Billing")).sum())
        assert result["data"]["count"] == expected

    def test_count_critical_unresolved(self, engine, df):
        from app.models.schemas import TicketFilters
        q = make_query(
            intent="COUNT_TICKETS",
            filters=TicketFilters(status="Unresolved", priority="Critical"),
        )
        result = engine.execute(q, df)
        expected = int((df["status"].isin(["Open", "Escalated"]) & (df["priority"] == "Critical")).sum())
        assert result["data"]["count"] == expected
        assert result["data"]["count"] == 31

    def test_list_critical_not_resolved_within_12_hours(self, engine, df):
        from app.models.schemas import TicketFilters
        q = make_query(
            intent="LIST_TICKETS",
            filters=TicketFilters(priority="Critical", not_resolved_within_hours=12),
        )
        result = engine.execute(q, df)
        assert len(result["data"]["rows"]) == 34

    def test_top_agent_resolved_defaults_to_most(self, engine, df):
        # Without any explicit sort in query, it should return the agent with the MOST resolved tickets
        q = make_query(intent="TOP_AGENT_RESOLVED")
        result = engine.execute(q, df)
        assert result["data"]["agent_id"] == "AGT-09"
        assert result["data"]["resolved_count"] == 37

    def test_count_empty_result(self, engine, df):
        from app.models.schemas import TicketFilters
        q = make_query(
            intent="COUNT_TICKETS",
            filters=TicketFilters(status="Resolved", priority="Critical", category="Billing"),
        )
        result = engine.execute(q, df)
        assert "count" in result["data"]
        assert isinstance(result["data"]["count"], int)


class TestAverageMetric:
    def test_avg_response_time(self, engine, df):
        from app.models.schemas import MetricField, AggregationOperation
        q = make_query(
            intent="AVERAGE_METRIC",
            metric=MetricField.RESPONSE_TIME,
            aggregation=AggregationOperation.AVERAGE,
        )
        result = engine.execute(q, df)
        expected = round(df["response_time_hrs"].mean(), 2)
        assert abs(result["data"]["average"] - expected) < 0.01

    def test_avg_rating_technical(self, engine, df):
        from app.models.schemas import MetricField, AggregationOperation, TicketFilters
        q = make_query(
            intent="AVERAGE_METRIC",
            metric=MetricField.CUSTOMER_RATING,
            aggregation=AggregationOperation.AVERAGE,
            filters=TicketFilters(category="Technical"),
        )
        result = engine.execute(q, df)
        expected = round(
            df[df["category"] == "Technical"]["customer_rating"].dropna().mean(), 2
        )
        assert abs(result["data"]["average"] - expected) < 0.01

    def test_avg_response_high_priority(self, engine, df):
        from app.models.schemas import MetricField, AggregationOperation, TicketFilters
        q = make_query(
            intent="AVERAGE_METRIC",
            metric=MetricField.RESPONSE_TIME,
            aggregation=AggregationOperation.AVERAGE,
            filters=TicketFilters(priority="High"),
        )
        result = engine.execute(q, df)
        expected = round(
            df[df["priority"] == "High"]["response_time_hrs"].mean(), 2
        )
        assert abs(result["data"]["average"] - expected) < 0.01


class TestListTickets:
    def test_list_returns_rows(self, engine, df):
        from app.models.schemas import TicketFilters
        q = make_query(intent="LIST_TICKETS", filters=TicketFilters(priority="Critical"))
        result = engine.execute(q, df)
        assert "rows" in result["data"]
        assert len(result["data"]["rows"]) > 0

    def test_list_respects_limit(self, engine, df):
        q = make_query(intent="LIST_TICKETS", limit=5)
        result = engine.execute(q, df)
        assert len(result["data"]["rows"]) <= 5

    def test_list_empty_filter(self, engine, df):
        from app.models.schemas import TicketFilters
        # There are no AGT-99 agents
        q = make_query(intent="LIST_TICKETS", filters=TicketFilters(agent_id="AGT-99"))
        result = engine.execute(q, df)
        assert result["data"]["rows"] == []


class TestGroupBy:
    def test_group_by_category(self, engine, df):
        from app.models.schemas import GroupByField, AggregationOperation
        q = make_query(
            intent="GROUP_BY",
            group_by=GroupByField.CATEGORY,
            aggregation=AggregationOperation.COUNT,
        )
        result = engine.execute(q, df)
        rows = result["data"]["rows"]
        assert len(rows) == 3  # General, Billing, Technical
        total = sum(r["count"] for r in rows)
        assert total == 500

    def test_group_by_priority(self, engine, df):
        from app.models.schemas import GroupByField, AggregationOperation
        q = make_query(
            intent="GROUP_BY",
            group_by=GroupByField.PRIORITY,
            aggregation=AggregationOperation.COUNT,
        )
        result = engine.execute(q, df)
        rows = result["data"]["rows"]
        assert len(rows) == 4  # Low, Medium, High, Critical


class TestRanking:
    def test_top_agent_resolved(self, engine, df):
        q = make_query(intent="TOP_AGENT_RESOLVED")
        result = engine.execute(q, df)
        assert "agent_id" in result["data"]
        assert "resolved_count" in result["data"]
        assert result["data"]["resolved_count"] > 0

    def test_top_agent_rating_lowest(self, engine, df):
        from app.models.schemas import MetricField, SortSpec, SortDirection
        q = make_query(
            intent="TOP_AGENT_RATING",
            metric=MetricField.CUSTOMER_RATING,
            sort=SortSpec(field="customer_rating", direction=SortDirection.ASC),
        )
        result = engine.execute(q, df)
        assert "agent_id" in result["data"]
        assert 1.0 <= result["data"]["avg_rating"] <= 5.0

    def test_top_category(self, engine, df):
        q = make_query(intent="TOP_CATEGORY")
        result = engine.execute(q, df)
        assert "category" in result["data"]
        assert "count" in result["data"]

    def test_unsupported_intent_handled(self, engine, df):
        q = make_query(intent="UNSUPPORTED", unsupported_reason="Test unsupported.")
        result = engine.execute(q, df)
        assert "answer" in result
        assert "unsupported" in result["answer"].lower() or "couldn't" in result["answer"].lower()
