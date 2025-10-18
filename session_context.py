"""Session-Kontext für Datum/Zeit.

Zentralisiert die Ermittlung eines Session-Basisdatums, damit relative
Zeitangaben ("morgen", "nächste Woche") konsistent interpretiert werden.

Verwendung:
    from session_context import init_session_time, get_session_date, get_session_datetime
    init_session_time()  # beim Start genau einmal
    heute = get_session_date()

Die Zeitzone ist standardmäßig Europe/Berlin. Fallback ist naive lokale Zeit,
falls ZoneInfo nicht verfügbar ist.
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _TZ = ZoneInfo("Europe/Berlin")
except Exception:  # pragma: no cover
    _TZ = None

_SESSION_DT: Optional[datetime] = None


def init_session_time(force: bool = False) -> datetime:
    """Initialisiert (oder erzwingt bei force=True neu) das Session-Basisdatum.

    Gibt den gesetzten datetime zurück.
    """
    global _SESSION_DT
    if _SESSION_DT is None or force:
        try:
            now_dt = datetime.now(_TZ) if _TZ else datetime.now()
        except Exception:
            now_dt = datetime.now()
        _SESSION_DT = now_dt
    return _SESSION_DT


def get_session_datetime() -> datetime:
    """Gibt den Session-Basis-datetime zurück (initialisiert falls nötig)."""
    if _SESSION_DT is None:
        return init_session_time()
    return _SESSION_DT


def get_session_date() -> date:
    """Gibt das Session-Basis-Datum zurück (initialisiert falls nötig)."""
    return get_session_datetime().date()


__all__ = [
    "init_session_time",
    "get_session_datetime",
    "get_session_date",
]
