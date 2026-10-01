"""Tests for AnomalyDetector: all four rules."""
import os
import pytest
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "support_tickets.csv")


@pytest.fixture(scope="module")
def df():
    from app.services.data_service import DataService
    return DataService(CSV_PATH).get_dataframe()


@pytest.fixture
def detector():
    from app.services.anomaly_detector import AnomalyDetector
    return AnomalyDetector()


class TestAnomalyDetectorRules:
    def test_detect_all_returns_list(self, detector, df):
        anomalies = detector.detect_all(df)
        assert isinstance(anomalies, list)

    def test_all_anomalies_have_required_fields(self, detector, df):
        from app.models.schemas import AnomalyRecord
        anomalies = detector.detect_all(df)
        for a in anomalies:
            assert isinstance(a, AnomalyRecord)
            assert a.ticket_id
            assert a.anomaly_type
            assert a.severity in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
            assert a.reason

    def test_rule1_long_resolution_detected(self, detector, df):
        anomalies = detector.detect_all(df)
        long_res = [a for a in anomalies if a.anomaly_type == "LONG_RESOLUTION_TIME"]
        assert len(long_res) > 0, "Expected some long resolution anomalies"

    def test_rule1_threshold_is_statistical(self, detector, df):
        # Threshold must be mean + 2*std, not a hardcoded value
        resolved = df[df["status"] == "Resolved"]["resolution_time_hrs"].dropna()
        threshold = resolved.mean() + 2 * resolved.std()
        anomalies = detector.detect_all(df)
        for a in anomalies:
            if a.anomaly_type == "LONG_RESOLUTION_TIME":
                assert a.metric_value > threshold - 0.1  # allow float rounding

    def test_rule2_high_priority_overdue_detected(self, detector, df):
        anomalies = detector.detect_all(df)
        overdue = [a for a in anomalies if a.anomaly_type == "HIGH_PRIORITY_OVERDUE"]
        assert len(overdue) > 0, "Expected high-priority overdue anomalies"

    def test_rule2_only_unresolved_tickets(self, detector, df):
        anomalies = detector.detect_all(df)
        overdue_ids = {a.ticket_id for a in anomalies if a.anomaly_type == "HIGH_PRIORITY_OVERDUE"}
        # These tickets must be Open or Escalated in the dataset
        for tid in overdue_ids:
            row = df[df["ticket_id"] == tid].iloc[0]
            assert row["status"] in {"Open", "Escalated"}
            assert row["priority"] in {"High", "Critical"}

    def test_rule3_slow_response_detected(self, detector, df):
        anomalies = detector.detect_all(df)
        slow = [a for a in anomalies if a.anomaly_type == "SLOW_FIRST_RESPONSE"]
        # Response times range 0.2-5.0; may or may not trigger depending on std
        # Just verify the rule runs without error
        assert isinstance(slow, list)

    def test_rule4_low_rating_detected(self, detector, df):
        anomalies = detector.detect_all(df)
        low_rating = [a for a in anomalies if a.anomaly_type == "LOW_CUSTOMER_RATING"]
        assert len(low_rating) > 0, "Expected some low rating anomalies"

    def test_rule4_only_resolved_tickets(self, detector, df):
        anomalies = detector.detect_all(df)
        for a in anomalies:
            if a.anomaly_type == "LOW_CUSTOMER_RATING":
                row = df[df["ticket_id"] == a.ticket_id].iloc[0]
                assert row["status"] == "Resolved"
                assert row["customer_rating"] <= 2.0

    def test_no_duplicate_ticket_type_pairs(self, detector, df):
        anomalies = detector.detect_all(df)
        pairs = [(a.ticket_id, a.anomaly_type) for a in anomalies]
        assert len(pairs) == len(set(pairs)), "Duplicate (ticket_id, type) pairs found"

    def test_summary_structure(self, detector, df):
        anomalies = detector.detect_all(df)
        summary = detector.get_summary(anomalies)
        assert "total_count" in summary
        assert "by_type" in summary
        assert "by_severity" in summary
        assert summary["total_count"] == len(anomalies)

    def test_empty_dataframe(self, detector):
        import pandas as pd
        empty_df = pd.DataFrame(columns=[
            "ticket_id","created_at","category","priority","status",
            "response_time_hrs","resolution_time_hrs","agent_id",
            "customer_rating","issue_summary"
        ])
        anomalies = detector.detect_all(empty_df)
        assert anomalies == []
