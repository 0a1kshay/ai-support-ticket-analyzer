"""
API integration tests using FastAPI TestClient.
LLM is mocked so tests run without a live LLM provider.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure the data path resolves correctly when running tests from any directory
os.environ["CSV_PATH"] = os.path.join(
    os.path.dirname(__file__), "..", "data", "support_tickets.csv"
)


def get_app():
    # Reset modules to get a fresh app
    import importlib
    import app.main as main_module
    importlib.reload(main_module)
    return main_module.app


@pytest.fixture(scope="module")
def client():
    """Create test client with a mocked LLM provider."""
    from app.main import create_app
    test_app = create_app()

    mock_llm = MagicMock()
    mock_llm.name = "MockLLM (test)"
    mock_llm.is_available.return_value = True

    with TestClient(test_app) as c:
        # Inject mock LLM into app state
        test_app.state.llm_provider = mock_llm
        from app.services.llm_service import IntentParser
        test_app.state.intent_parser = IntentParser(mock_llm)
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_structure(self, client):
        r = client.get("/health")
        d = r.json()
        assert "status" in d
        assert "dataset_loaded" in d
        assert "total_rows" in d
        assert "llm_provider" in d
        assert "llm_available" in d
        assert "message" in d

    def test_health_dataset_loaded(self, client):
        r = client.get("/health")
        assert r.json()["dataset_loaded"] is True
        assert r.json()["total_rows"] == 500


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200

    def test_stats_structure(self, client):
        r = client.get("/api/stats")
        d = r.json()
        required = [
            "total_tickets", "open_tickets", "resolved_tickets",
            "escalated_tickets", "critical_tickets",
            "avg_response_time_hrs", "avg_resolution_time_hrs",
            "avg_customer_rating", "anomaly_count",
            "categories", "priorities", "top_agents_by_resolved",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_stats_total_is_500(self, client):
        r = client.get("/api/stats")
        assert r.json()["total_tickets"] == 500

    def test_stats_counts_sum(self, client):
        r = client.get("/api/stats")
        d = r.json()
        assert (
            d["open_tickets"] + d["resolved_tickets"] + d["escalated_tickets"]
            == d["total_tickets"]
        )


class TestAnomaliesEndpoint:
    def test_anomalies_returns_200(self, client):
        r = client.get("/api/anomalies")
        assert r.status_code == 200

    def test_anomalies_structure(self, client):
        r = client.get("/api/anomalies")
        d = r.json()
        assert "total_count" in d
        assert "by_type" in d
        assert "by_severity" in d
        assert "anomalies" in d
        assert isinstance(d["anomalies"], list)

    def test_anomalies_non_empty(self, client):
        r = client.get("/api/anomalies")
        d = r.json()
        assert d["total_count"] > 0

    def test_anomaly_records_have_required_fields(self, client):
        r = client.get("/api/anomalies")
        for a in r.json()["anomalies"][:5]:
            assert "ticket_id" in a
            assert "anomaly_type" in a
            assert "severity" in a
            assert "reason" in a

    def test_anomalies_filter_by_severity(self, client):
        r = client.get("/api/anomalies?severity=HIGH")
        d = r.json()
        for a in d["anomalies"]:
            assert a["severity"] == "HIGH"

    def test_anomalies_filter_by_type(self, client):
        r = client.get("/api/anomalies?anomaly_type=LOW_CUSTOMER_RATING")
        d = r.json()
        for a in d["anomalies"]:
            assert a["anomaly_type"] == "LOW_CUSTOMER_RATING"


class TestQueryEndpoint:
    def _mock_llm_response(self, client, json_str: str):
        """Helper to set what the mock LLM returns."""
        client.app.state.llm_provider.complete.return_value = json_str
        from app.services.llm_service import IntentParser
        client.app.state.intent_parser = IntentParser(client.app.state.llm_provider)

    def test_query_empty_question_rejected(self, client):
        r = client.post("/api/query", json={"question": "   "})
        # Should return 422 (validation error) or 200 with error message
        assert r.status_code in (422, 200)

    def test_query_count_tickets(self, client):
        self._mock_llm_response(client, '''{
            "intent":"COUNT_TICKETS",
            "filters":{"status":"Open","priority":null,"category":null,"agent_id":null,
                       "older_than_hours":null,"min_resolution_hours":null,
                       "max_resolution_hours":null,"min_response_hours":null,
                       "max_response_hours":null,"min_rating":null,"max_rating":null},
            "metric":null,"aggregation":null,"group_by":null,
            "sort":null,"limit":null,"date_filter":null,"unsupported_reason":null
        }''')
        r = client.post("/api/query", json={"question": "How many tickets are open?"})
        assert r.status_code == 200
        d = r.json()
        assert d["intent"] == "COUNT_TICKETS"
        assert "count" in d["data"]
        assert d["data"]["count"] > 0

    def test_query_average_metric(self, client):
        self._mock_llm_response(client, '''{
            "intent":"AVERAGE_METRIC",
            "filters":{"status":null,"priority":"High","category":null,"agent_id":null,
                       "older_than_hours":null,"min_resolution_hours":null,
                       "max_resolution_hours":null,"min_response_hours":null,
                       "max_response_hours":null,"min_rating":null,"max_rating":null},
            "metric":"response_time_hrs","aggregation":"average","group_by":null,
            "sort":null,"limit":null,"date_filter":null,"unsupported_reason":null
        }''')
        r = client.post("/api/query", json={"question": "Avg response time for High priority?"})
        assert r.status_code == 200
        d = r.json()
        assert d["intent"] == "AVERAGE_METRIC"
        assert "average" in d["data"]

    def test_query_unsupported_intent(self, client):
        self._mock_llm_response(client, '''{
            "intent":"UNSUPPORTED",
            "filters":null,"metric":null,"aggregation":null,"group_by":null,
            "sort":null,"limit":null,"date_filter":null,
            "unsupported_reason":"This question is not about support tickets."
        }''')
        r = client.post("/api/query", json={"question": "What is the weather?"})
        assert r.status_code == 200
        d = r.json()
        assert d["intent"] == "UNSUPPORTED"

    def test_query_invalid_llm_json(self, client):
        client.app.state.llm_provider.complete.return_value = "not json at all"
        from app.services.llm_service import IntentParser
        client.app.state.intent_parser = IntentParser(client.app.state.llm_provider)
        r = client.post("/api/query", json={"question": "Something?"})
        assert r.status_code == 200  # graceful error
        d = r.json()
        assert d["intent"] in ("PARSE_ERROR", "LLM_UNAVAILABLE", "VALIDATION_ERROR", "ERROR")

    def test_query_llm_unavailable(self, client):
        from app.services.llm_service import LLMUnavailableError
        client.app.state.llm_provider.complete.side_effect = LLMUnavailableError("offline")
        from app.services.llm_service import IntentParser
        client.app.state.intent_parser = IntentParser(client.app.state.llm_provider)
        r = client.post("/api/query", json={"question": "How many open tickets?"})
        assert r.status_code == 200
        d = r.json()
        assert d["intent"] == "LLM_UNAVAILABLE"
        # Reset side effect
        client.app.state.llm_provider.complete.side_effect = None

    def test_query_response_structure(self, client):
        self._mock_llm_response(client, '''{
            "intent":"COUNT_TICKETS",
            "filters":{"status":null,"priority":null,"category":null,"agent_id":null,
                       "older_than_hours":null,"min_resolution_hours":null,
                       "max_resolution_hours":null,"min_response_hours":null,
                       "max_response_hours":null,"min_rating":null,"max_rating":null},
            "metric":null,"aggregation":null,"group_by":null,
            "sort":null,"limit":null,"date_filter":null,"unsupported_reason":null
        }''')
        r = client.post("/api/query", json={"question": "How many total tickets?"})
        assert r.status_code == 200
        d = r.json()
        assert "question" in d
        assert "intent" in d
        assert "answer" in d
        assert "data" in d


class TestUIEndpoints:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_docs_returns_200(self, client):
        r = client.get("/docs")
        assert r.status_code == 200
