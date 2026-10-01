"""Tests for DataService: loading, validation, normalisation."""
import os
import pytest
import pandas as pd

# Point to the actual CSV in the data/ folder
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "support_tickets.csv")


def get_service():
    from app.services.data_service import DataService
    return DataService(CSV_PATH)


class TestDataServiceLoad:
    def test_csv_loads_without_error(self):
        svc = get_service()
        assert svc.is_loaded

    def test_row_count_is_500(self):
        svc = get_service()
        assert svc.row_count == 500

    def test_required_columns_present(self):
        from app.services.data_service import REQUIRED_COLUMNS
        svc = get_service()
        df = svc.get_dataframe()
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_created_at_is_datetime(self):
        svc = get_service()
        df = svc.get_dataframe()
        assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    def test_no_null_created_at(self):
        svc = get_service()
        df = svc.get_dataframe()
        assert df["created_at"].notna().all()

    def test_numeric_columns(self):
        svc = get_service()
        df = svc.get_dataframe()
        for col in ("response_time_hrs", "resolution_time_hrs", "customer_rating"):
            assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"

    def test_missing_resolution_and_rating_preserved(self):
        svc = get_service()
        df = svc.get_dataframe()
        # Dataset has 173 nulls in each
        assert df["resolution_time_hrs"].isna().sum() > 0
        assert df["customer_rating"].isna().sum() > 0

    def test_valid_statuses(self):
        svc = get_service()
        df = svc.get_dataframe()
        valid = {"Open", "Resolved", "Escalated"}
        assert set(df["status"].unique()).issubset(valid)

    def test_valid_priorities(self):
        svc = get_service()
        df = svc.get_dataframe()
        valid = {"Low", "Medium", "High", "Critical"}
        assert set(df["priority"].unique()).issubset(valid)

    def test_valid_categories(self):
        svc = get_service()
        df = svc.get_dataframe()
        valid = {"General", "Billing", "Technical"}
        assert set(df["category"].unique()).issubset(valid)

    def test_get_stats_returns_dict(self):
        svc = get_service()
        stats = svc.get_stats()
        required_keys = [
            "total_tickets", "open_tickets", "resolved_tickets",
            "escalated_tickets", "critical_tickets",
            "avg_response_time_hrs", "avg_resolution_time_hrs",
            "avg_customer_rating", "categories", "priorities",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_counts_sum_to_total(self):
        svc = get_service()
        stats = svc.get_stats()
        total = stats["total_tickets"]
        assert (
            stats["open_tickets"] + stats["resolved_tickets"] + stats["escalated_tickets"]
            == total
        )

    def test_missing_csv_raises_error(self):
        from app.services.data_service import DataService, DataServiceError
        with pytest.raises(DataServiceError):
            DataService("/nonexistent/path/file.csv")

    def test_schema_returns_dict(self):
        svc = get_service()
        schema = svc.get_schema()
        assert isinstance(schema, dict)
        assert "ticket_id" in schema
