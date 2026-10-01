"""
Deterministic Anomaly Detector.

Rules (all thresholds are computed statistically from the data at runtime;
nothing is hardcoded to the specific sample values):

  Rule 1 – LONG_RESOLUTION_TIME
    Threshold: mean + 2 * std of resolution_time_hrs for resolved tickets.
    Flag resolved tickets above this threshold.

  Rule 2 – HIGH_PRIORITY_UNRESOLVED_OVERDUE
    Flag Open/Escalated tickets with priority High or Critical
    whose age (relative to dataset reference date) > 24 hours.

  Rule 3 – SLOW_FIRST_RESPONSE
    Threshold: mean + 2 * std of response_time_hrs across all tickets.
    Flag tickets with response_time_hrs above this threshold.

  Rule 4 – LOW_CUSTOMER_RATING
    Flag resolved tickets with customer_rating <= 2.

The LLM is NOT used here. All logic is deterministic Python/Pandas.
"""
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from app.models.schemas import AnomalyRecord

logger = logging.getLogger(__name__)

DATASET_REFERENCE_DATE = pd.Timestamp("2024-03-30 23:59:59")


class AnomalyDetector:
    """Computes anomalies from the DataFrame on every call (fast for 500 rows)."""

    def detect_all(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        """Run all four rules and return deduplicated anomaly records."""
        anomalies: list[AnomalyRecord] = []
        seen: set[tuple[str, str]] = set()

        for rule in (
            self._rule_long_resolution,
            self._rule_high_priority_overdue,
            self._rule_slow_response,
            self._rule_low_rating,
        ):
            try:
                for a in rule(df):
                    key = (a.ticket_id, a.anomaly_type)
                    if key not in seen:
                        seen.add(key)
                        anomalies.append(a)
            except Exception as exc:
                logger.error("Anomaly rule %s failed: %s", rule.__name__, exc)

        return anomalies

    # ------------------------------------------------------------------
    # Rule 1: Long resolution time
    # ------------------------------------------------------------------

    def _rule_long_resolution(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        resolved = df[df["status"] == "Resolved"].copy()
        valid = resolved[resolved["resolution_time_hrs"].notna()]
        if len(valid) < 5:
            return []

        mean = valid["resolution_time_hrs"].mean()
        std = valid["resolution_time_hrs"].std()
        threshold = round(mean + 2 * std, 2)

        flagged = valid[valid["resolution_time_hrs"] > threshold]
        results = []
        for _, row in flagged.iterrows():
            val = round(float(row["resolution_time_hrs"]), 2)
            results.append(AnomalyRecord(
                ticket_id=row["ticket_id"],
                anomaly_type="LONG_RESOLUTION_TIME",
                severity="HIGH" if val > mean + 3 * std else "MEDIUM",
                reason=(
                    f"Resolution time of {val}h exceeds the statistical "
                    f"threshold of {threshold}h (mean={round(mean,2)}h + 2*std)."
                ),
                metric_value=val,
                threshold=threshold,
            ))
        logger.info(
            "Rule 1 (LONG_RESOLUTION_TIME): threshold=%.2fh, flagged=%d", threshold, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Rule 2: High-priority unresolved older than 24 hours
    # ------------------------------------------------------------------

    def _rule_high_priority_overdue(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        threshold_hours = 24.0
        unresolved = df[
            (df["status"].isin(["Open", "Escalated"])) &
            (df["priority"].isin(["High", "Critical"])) &
            (df["created_at"].notna())
        ].copy()

        if unresolved.empty:
            return []

        unresolved["age_hours"] = (
            (DATASET_REFERENCE_DATE - unresolved["created_at"])
            .dt.total_seconds() / 3600
        )
        flagged = unresolved[unresolved["age_hours"] > threshold_hours]

        results = []
        for _, row in flagged.iterrows():
            age = round(float(row["age_hours"]), 1)
            sev = "CRITICAL" if row["priority"] == "Critical" else "HIGH"
            results.append(AnomalyRecord(
                ticket_id=row["ticket_id"],
                anomaly_type="HIGH_PRIORITY_OVERDUE",
                severity=sev,
                reason=(
                    f"{row['priority']} priority ticket is {row['status']} "
                    f"and is {age}h old (threshold: {threshold_hours}h)."
                ),
                metric_value=age,
                threshold=threshold_hours,
            ))
        logger.info(
            "Rule 2 (HIGH_PRIORITY_OVERDUE): threshold=%.1fh, flagged=%d",
            threshold_hours, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Rule 3: Slow first response
    # ------------------------------------------------------------------

    def _rule_slow_response(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        valid = df[df["response_time_hrs"].notna()].copy()
        if len(valid) < 5:
            return []

        mean = valid["response_time_hrs"].mean()
        std = valid["response_time_hrs"].std()
        threshold = round(mean + 2 * std, 2)

        flagged = valid[valid["response_time_hrs"] > threshold]
        results = []
        for _, row in flagged.iterrows():
            val = round(float(row["response_time_hrs"]), 2)
            results.append(AnomalyRecord(
                ticket_id=row["ticket_id"],
                anomaly_type="SLOW_FIRST_RESPONSE",
                severity="MEDIUM",
                reason=(
                    f"Response time of {val}h exceeds the statistical "
                    f"threshold of {threshold}h (mean={round(mean,2)}h + 2*std)."
                ),
                metric_value=val,
                threshold=threshold,
            ))
        logger.info(
            "Rule 3 (SLOW_FIRST_RESPONSE): threshold=%.2fh, flagged=%d", threshold, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Rule 4: Low customer rating
    # ------------------------------------------------------------------

    def _rule_low_rating(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        threshold = 2.0
        flagged = df[
            (df["status"] == "Resolved") &
            (df["customer_rating"].notna()) &
            (df["customer_rating"] <= threshold)
        ]
        results = []
        for _, row in flagged.iterrows():
            rating = float(row["customer_rating"])
            results.append(AnomalyRecord(
                ticket_id=row["ticket_id"],
                anomaly_type="LOW_CUSTOMER_RATING",
                severity="HIGH" if rating == 1.0 else "MEDIUM",
                reason=(
                    f"Resolved ticket has a customer rating of {rating}/5 "
                    f"(threshold: <= {threshold}/5)."
                ),
                metric_value=rating,
                threshold=threshold,
            ))
        logger.info("Rule 4 (LOW_CUSTOMER_RATING): flagged=%d", len(results))
        return results

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def get_summary(self, anomalies: list[AnomalyRecord]) -> dict:
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for a in anomalies:
            by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
        return {
            "total_count": len(anomalies),
            "by_type": by_type,
            "by_severity": by_severity,
        }
