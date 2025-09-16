"""Datumshilfen.

adjust_dates_if_year_missing(user_text, start_iso, end_iso) -> (start_iso, end_iso, adjusted: bool)
Logik:
 - Wenn im user_text keine 4-stellige Jahreszahl vorkommt
 - UND start_datetime klar in der Vergangenheit liegt (mehr als 10 Minuten vor jetzt)
   -> addiere 1 Jahr (rollover) für start & end (gleicher Offset)
 - Leap-Day (29. Feb) wird bei fehlender Existenz im nächsten Jahr auf 28. Feb gelegt.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import re

_DEF_TOLERANCE_MIN = 10  # Minuten

_year_pattern = re.compile(r"\b\d{4}\b")

def _safe_add_year(dt: datetime) -> datetime:
    try:
        return dt.replace(year=dt.year + 1)
    except ValueError:
        # 29. Feb -> 28. Feb
        if dt.month == 2 and dt.day == 29:
            return dt.replace(year=dt.year + 1, day=28)
        raise


def adjust_dates_if_year_missing(user_text: str, start_iso: str, end_iso: str):
    try:
        s = datetime.fromisoformat(start_iso)
        e = datetime.fromisoformat(end_iso)
    except Exception:
        return start_iso, end_iso, False
    if _year_pattern.search(user_text):
        return start_iso, end_iso, False
    now = datetime.now(s.tzinfo) if s.tzinfo else datetime.now()
    if s < now - timedelta(minutes=_DEF_TOLERANCE_MIN):
        new_s = _safe_add_year(s)
        # Dauer beibehalten
        delta = e - s
        new_e = new_s + delta
        return new_s.isoformat(), new_e.isoformat(), True
    return start_iso, end_iso, False

__all__ = ["adjust_dates_if_year_missing"]
