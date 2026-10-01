"""
Utility helpers: date parsing, rounding, safe division, etc.
"""
from __future__ import annotations
import math
import logging
from datetime import datetime, timedelta, date
from typing import Optional

logger = logging.getLogger(__name__)


def round_2(value: Optional[float]) -> Optional[float]:
    """Round to 2 decimal places or return None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 2)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def parse_date_filter(filter_type: str,
                      date_from: Optional[str] = None,
                      date_to: Optional[str] = None,
                      reference_date: Optional[date] = None):
    """
    Returns (start_dt, end_dt) as pandas-compatible timestamps.

    NOTE on dataset temporal context:
    The dataset contains tickets from 2024-01-01 to 2024-03-30.
    Relative terms like 'this week' are resolved against the dataset's
    most recent date (or a provided reference_date) so historical data
    is handled sensibly rather than comparing against today's wall-clock date.
    This behaviour is documented in the README.
    """
    from pandas import Timestamp

    ref = reference_date or date(2024, 3, 30)  # dataset end date as default ref

    if filter_type == "today":
        start = datetime.combine(ref, datetime.min.time())
        end = datetime.combine(ref, datetime.max.time())
    elif filter_type == "yesterday":
        d = ref - timedelta(days=1)
        start = datetime.combine(d, datetime.min.time())
        end = datetime.combine(d, datetime.max.time())
    elif filter_type == "this_week":
        monday = ref - timedelta(days=ref.weekday())
        start = datetime.combine(monday, datetime.min.time())
        end = datetime.combine(monday + timedelta(days=6), datetime.max.time())
    elif filter_type == "last_week":
        monday = ref - timedelta(days=ref.weekday() + 7)
        start = datetime.combine(monday, datetime.min.time())
        end = datetime.combine(monday + timedelta(days=6), datetime.max.time())
    elif filter_type == "this_month":
        start = datetime(ref.year, ref.month, 1)
        if ref.month == 12:
            end = datetime(ref.year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(ref.year, ref.month + 1, 1) - timedelta(seconds=1)
    elif filter_type == "last_month":
        if ref.month == 1:
            start = datetime(ref.year - 1, 12, 1)
            end = datetime(ref.year, 1, 1) - timedelta(seconds=1)
        else:
            start = datetime(ref.year, ref.month - 1, 1)
            end = datetime(ref.year, ref.month, 1) - timedelta(seconds=1)
    elif filter_type == "after":
        start = datetime.fromisoformat(date_from) if date_from else datetime.min
        end = Timestamp.max.to_pydatetime()
    elif filter_type == "before":
        start = datetime.min
        end = datetime.fromisoformat(date_to) if date_to else datetime.max
    elif filter_type == "between":
        start = datetime.fromisoformat(date_from) if date_from else datetime.min
        end = datetime.fromisoformat(date_to) if date_to else datetime.max
    else:
        raise ValueError(f"Unknown date filter type: {filter_type}")

    return Timestamp(start), Timestamp(end)
