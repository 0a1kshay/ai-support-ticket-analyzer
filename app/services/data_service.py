"""
Data service: loads the CSV once at startup, validates, normalises, and
exposes clean accessor methods to the rest of the application.
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column name mapping from any variant the CSV may use → internal canonical name
COLUMN_ALIASES: dict[str, str] = {
    # Response time variants
    "resp_time_hrs": "response_time_hrs",
    "response_time": "response_time_hrs",
    # Resolution time variants
    "resol_time_hrs": "resolution_time_hrs",
    "resolution_time": "resolution_time_hrs",
    # Rating variants
    "cust_rating": "customer_rating",
    "rating": "customer_rating",
}

REQUIRED_COLUMNS = {
    "ticket_id", "created_at", "category", "priority",
    "status", "response_time_hrs", "resolution_time_hrs",
    "agent_id", "customer_rating", "issue_summary",
}

# Categorical allowlists for validation
VALID_CATEGORIES = {"General", "Billing", "Technical"}
VALID_PRIORITIES = {"Low", "Medium", "High", "Critical"}
VALID_STATUSES   = {"Open", "Resolved", "Escalated"}


class DataServiceError(Exception):
    """Raised when the dataset cannot be loaded or is structurally invalid."""


class DataService:
    """
    Singleton-style service that owns the in-memory DataFrame.
    Loaded once at application startup.
    """

    def __init__(self, csv_path: str):
        self._csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate_csv(self) -> str:
        if self._csv_path and os.path.isfile(self._csv_path):
            return self._csv_path
        if self._csv_path and self._csv_path not in ("data/support_tickets.csv", "support_tickets.csv", os.path.join("data", "support_tickets.csv")):
            raise DataServiceError(
                f"Dataset not found at specified path: {self._csv_path}"
            )
        # Try relative locations as fallback
        candidates = [
            os.path.join("data", "support_tickets.csv"),
            "support_tickets.csv",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise DataServiceError(
            f"Dataset not found. Tried: {candidates}. "
            "Place support_tickets.csv in the data/ directory."
        )

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename aliased column names to the canonical internal schema."""
        rename_map = {
            col: COLUMN_ALIASES[col]
            for col in df.columns
            if col in COLUMN_ALIASES
        }
        if rename_map:
            logger.info("Renaming columns: %s", rename_map)
            df = df.rename(columns=rename_map)
        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataServiceError(
                f"Dataset is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(df.columns)}"
            )

    def _load(self) -> None:
        path = self._locate_csv()
        logger.info("Loading dataset from: %s", path)

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise DataServiceError(f"Failed to read CSV: {exc}") from exc

        df = self._normalise_columns(df)
        self._validate_required_columns(df)

        # Parse datetime
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        bad_dates = df["created_at"].isna().sum()
        if bad_dates > 0:
            logger.warning("%d rows have unparseable created_at timestamps.", bad_dates)

        # Numeric conversion
        for col in ("response_time_hrs", "resolution_time_hrs", "customer_rating"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Categorical validation (log warnings; do not drop rows)
        for col, valid in [
            ("category", VALID_CATEGORIES),
            ("priority", VALID_PRIORITIES),
            ("status", VALID_STATUSES),
        ]:
            invalid_mask = ~df[col].isin(valid) & df[col].notna()
            if invalid_mask.any():
                bad_vals = df.loc[invalid_mask, col].unique().tolist()
                logger.warning(
                    "Column '%s' contains unrecognised values: %s", col, bad_vals
                )

        # Strip whitespace from string columns
        for col in ("ticket_id", "category", "priority", "status", "agent_id"):
            df[col] = df[col].astype(str).str.strip()

        self._df = df
        logger.info(
            "Dataset loaded: %d rows, %d columns. "
            "Missing resolution_time: %d, missing customer_rating: %d",
            len(df), len(df.columns),
            df["resolution_time_hrs"].isna().sum(),
            df["customer_rating"].isna().sum(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        """Return a copy of the loaded DataFrame."""
        return self._df.copy()

    def get_schema(self) -> dict:
        """Return column names and dtypes."""
        return {col: str(dtype) for col, dtype in self._df.dtypes.items()}

    def get_stats(self) -> dict:
        """Return summary statistics for dashboard KPIs."""
        df = self._df
        return {
            "total_tickets": len(df),
            "open_tickets": int((df["status"] == "Open").sum()),
            "resolved_tickets": int((df["status"] == "Resolved").sum()),
            "escalated_tickets": int((df["status"] == "Escalated").sum()),
            "critical_tickets": int((df["priority"] == "Critical").sum()),
            "avg_response_time_hrs": round(df["response_time_hrs"].mean(), 2),
            "avg_resolution_time_hrs": round(
                df["resolution_time_hrs"].dropna().mean(), 2
            ) if df["resolution_time_hrs"].notna().any() else None,
            "avg_customer_rating": round(
                df["customer_rating"].dropna().mean(), 2
            ) if df["customer_rating"].notna().any() else None,
            "categories": df["category"].value_counts().to_dict(),
            "priorities": df["priority"].value_counts().to_dict(),
            "top_agents_by_resolved": (
                df[df["status"] == "Resolved"]
                .groupby("agent_id")
                .size()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
                .rename(columns={0: "resolved_count"})
                .to_dict(orient="records")
            ),
        }

    @property
    def is_loaded(self) -> bool:
        return self._df is not None and len(self._df) > 0

    @property
    def row_count(self) -> int:
        return len(self._df) if self._df is not None else 0
