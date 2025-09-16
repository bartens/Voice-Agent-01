"""Google Calendar Tool-Implementierung für Function Calling.

Enthält create_calendar_event(...) das vom Gemini Function Call Wrapper genutzt wird.
"""
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any, Callable, TypeVar, Tuple
from datetime import datetime, timedelta

from config import (
    CALENDAR_ID,
    GOOGLE_OAUTH_CLIENT_SECRET,
    GOOGLE_OAUTH_TOKEN,
)

# Google API Imports erst beim Aufruf laden (lazy), um Startup ohne Kalender zu erlauben.
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Erweiterte Scope, damit freebusy (Verfügbarkeitsprüfung) funktioniert.
# HINWEIS: Falls bereits ein Token mit nur calendar.events existiert,
# bitte die Token-Datei löschen, um neue Berechtigungen zu erhalten.
_raw_scopes = os.getenv("GOOGLE_CALENDAR_SCOPES", "https://www.googleapis.com/auth/calendar")
SCOPES = [s.strip() for s in _raw_scopes.split(',') if s.strip()]
CLIENT_SECRET_FILE = GOOGLE_OAUTH_CLIENT_SECRET
TOKEN_FILE = GOOGLE_OAUTH_TOKEN

# Interner Cache für aufgelöste Kalender-ID, um Mehrfach-API Calls zu vermeiden
_RESOLVED_CAL_ID: str | None = None

def _resolve_calendar_id(service) -> str:
    """Ermittelt die tatsächlich zu verwendende Calendar ID.

    Regeln:
      - Falls ENV/Config CALENDAR_ID leer (""), "auto" oder "primary": Verwende den primären Kalender des authentifizierten Users.
      - Falls eine explizite ID gesetzt ist, nutze diese unverändert.
      - Ergebnis wird global gecached (bis Prozessende), erneute Aufrufe kostengünstig.

    Gibt bei Fehlern eine Exception aus, damit der Aufrufer entscheiden kann, wie er reagiert.
    """
    global _RESOLVED_CAL_ID
    if _RESOLVED_CAL_ID:
        return _RESOLVED_CAL_ID
    raw = (CALENDAR_ID or "").strip().lower()
    if raw in {"", "auto", "primary"}:
        # Primären Kalender abfragen
        try:
            me = service.calendarList().get(calendarId='primary').execute()
            _RESOLVED_CAL_ID = me.get('id') or 'primary'
            if os.getenv("CALENDAR_DEBUG", "0").lower() in {"1","true","yes","on"}:
                print(f"[CAL][DEBUG] AUTO resolved calendarId -> {_RESOLVED_CAL_ID}")
        except Exception as e:
            raise RuntimeError(f"Primärer Kalender konnte nicht ermittelt werden: {e}")
    else:
        _RESOLVED_CAL_ID = CALENDAR_ID
        if os.getenv("CALENDAR_DEBUG", "0").lower() in {"1","true","yes","on"}:
            print(f"[CAL][DEBUG] Using explicit calendarId -> {_RESOLVED_CAL_ID}")
    return _RESOLVED_CAL_ID


def _load_credentials():
    """Lädt oder erzeugt OAuth Credentials.

    Unterstützt erzwungenen Reauth via ENV FORCE_OAUTH_REAUTH=1.
    Gibt bei Problemen klare Exceptions aus, damit der Aufrufer Benutzerfreundliche Meldungen liefern kann.
    """
    force = os.getenv("FORCE_OAUTH_REAUTH", "0").lower() in {"1","true","yes","on"}
    creds = None
    if not force and os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"[CAL] Warn: Token-Datei unlesbar – ignoriere ({e})")
            creds = None
    if force:
        print("[CAL] FORCE_OAUTH_REAUTH aktiv – existierendes Token wird ignoriert.")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token and not force:
            try:
                creds.refresh(Request())
                print("[CAL] Refresh Token verwendet – kein Browser-Login nötig.")
            except Exception as e:
                print(f"[CAL] Refresh fehlgeschlagen: {e} – starte neuen Flow.")
                creds = None
        if not creds or not creds.valid:
            if not os.path.exists(CLIENT_SECRET_FILE):
                raise FileNotFoundError(f"Client Secret Datei nicht gefunden: {CLIENT_SECRET_FILE}")
            print("[CAL] Starte neuen OAuth Browser-Flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            print("[CAL] OAuth abgeschlossen – speichere Token.")
        try:
            with open(TOKEN_FILE, 'w') as f:
                f.write(creds.to_json())
        except Exception as e:
            print(f"[CAL] Warn: Token konnte nicht geschrieben werden: {e}")
    return creds


def _get_service():
    creds = _load_credentials()
    from google.oauth2.credentials import Credentials as _UserCreds
    if isinstance(creds, _UserCreds) and not creds.valid:
        raise RuntimeError("Keine gültigen Google Calendar Credentials vorhanden (OAuth).")
    try:
        ct = type(creds).__name__
        if os.getenv("CAL_LOG_CREDENTIAL_INFO_ONCE", "1") == "1":
            # Einmalige Ausgabe (primitive Guard via Env Reset)
            print(f"[CAL] Verwende Credential-Typ: {ct}")
            os.environ["CAL_LOG_CREDENTIAL_INFO_ONCE"] = "0"
    except Exception:
        pass
    return build("calendar", "v3", credentials=creds, cache_discovery=False)

def pre_auth() -> bool:
    """Versucht vorab den Auth Flow auszulösen.
    Rückgabe True wenn gültige Credentials vorhanden/erstellt, sonst False.
    """
    try:
        _get_service()
        return True
    except Exception as e:
        print(f"[CAL] Pre-Auth fehlgeschlagen: {e}")
        return False


T = TypeVar('T')

def _with_retries(fn: Callable[[], T], *, retries: int = 3, base_delay: float = 0.7, factor: float = 2.0) -> T:
    """Führt einen API-Call mit exponentiellem Backoff aus.

    Retry bei typischen transienten Fehlern (429, 500, 502, 503, 504). Andere Fehler werden direkt propagiert.
    """
    import random, time as _t
    last_err: Exception | None = None
    for attempt in range(1, retries+1):
        try:
            return fn()
        except HttpError as he:  # Typische Google API Fehlerstruktur
            status = getattr(he, 'status_code', None) or getattr(he.resp, 'status', None)
            if status in {429, 500, 502, 503, 504} and attempt < retries:
                delay = base_delay * (factor ** (attempt-1)) * (1 + random.random()*0.25)
                if os.getenv("CALENDAR_DEBUG", "0").lower() in {"1","true","yes","on"}:
                    print(f"[CAL][RETRY] Attempt {attempt} failed (status={status}) – retry in {delay:.2f}s")
                _t.sleep(delay)
                last_err = he
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                delay = base_delay * (factor ** (attempt-1))
                if os.getenv("CALENDAR_DEBUG", "0").lower() in {"1","true","yes","on"}:
                    print(f"[CAL][RETRY] Attempt {attempt} exception: {e} – retrying in {delay:.2f}s")
                _t.sleep(delay)
                continue
            raise
    # Falls hier (eigentlich unreachable)
    assert last_err is not None
    raise last_err

def _normalize_dt(raw: str) -> str:
    """Versucht verschiedene leicht fehlerhafte Eingabeformate in RFC3339 (ohne Offset) zu normalisieren.

    Beispiele:
      2025-09-29 15:00  -> 2025-09-29T15:00:00
      2025-09-29T1500    -> 2025-09-29T15:00:00
      2025-09-29 1500    -> 2025-09-29T15:00:00
      2025-09-29T15:00   -> 2025-09-29T15:00:00
    Falls bereits Sekunden vorhanden -> unverändert.
    Offset/ Zeitzonen-Suffixe bleiben erhalten.
    """
    r = raw.strip()
    if 'T' not in r and ' ' in r:
        # ersetze erstes Leerzeichen durch T
        parts = r.split()
        if len(parts) == 2:
            r = parts[0] + 'T' + parts[1]
    # Wenn kein ':' im Zeitteil oder nur einmal -> ergänzen
    if 'T' in r:
        date_part, time_part = r.split('T', 1)
        # Entferne evtl. Offset für Bearbeitung
        offset = ''
        for sep in ['+', '-']:
            if sep in time_part[1:]:  # nicht erstes Zeichen (negatives) berücksichtigen
                base, off = time_part.split(sep, 1)
                offset = sep + off
                time_part = base
                break
        if 'Z' in time_part:
            base, zrest = time_part.split('Z', 1)
            offset = 'Z' + zrest
            time_part = base
        digits = time_part.replace(':', '')
        if digits.isdigit():
            if len(digits) == 4:  # HHMM
                time_part = digits[:2] + ':' + digits[2:] + ':00'
            elif len(digits) == 2:  # HH -> add :00:00
                time_part = digits + ':00:00'
            elif len(digits) == 6:  # HHMMSS
                time_part = digits[:2] + ':' + digits[2:4] + ':' + digits[4:]
        # Falls nur HH:MM
        if time_part.count(':') == 1:
            time_part = time_part + ':00'
        r = date_part + 'T' + time_part + offset
    return r


def create_calendar_event(summary: str,
                          start_iso: str,
                          end_iso: str,
                          timezone: str = "Europe/Berlin",
                          description: Optional[str] = None,
                          attendees: Optional[List[str]] = None,
                          location: Optional[str] = None,
                          recurrence: Optional[List[str]] = None,
                          reminders_override_minutes: Optional[List[int]] = None,
                          use_default_reminders: bool | None = None) -> Dict[str, Any]:
    """Legt ein Kalender-Event an und liefert Kerninformationen zurück.

    Erwartet ISO 8601 Strings (YYYY-MM-DDTHH:MM:SS oder mit Zeitzone). Falls keine Zeitzone
    enthalten ist, wird timezone genutzt.
    """
    # Validierung einfache Zeitlogik
    # Normalisieren bevor wir parsen
    start_iso_norm = _normalize_dt(start_iso)
    end_iso_norm = _normalize_dt(end_iso)
    # Falls keine Zeitzoneninfo vorhanden, mit gewünschter timezone versehen und als RFC3339 mit Offset serialisieren
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        tz = ZoneInfo(timezone)
    except Exception:
        tz = None
    try:
        s_dt = datetime.fromisoformat(start_iso_norm.replace('Z', '+00:00'))
        e_dt = datetime.fromisoformat(end_iso_norm.replace('Z', '+00:00'))
        if s_dt.tzinfo is None and tz:
            s_dt = s_dt.replace(tzinfo=tz)
        if e_dt.tzinfo is None and tz:
            e_dt = e_dt.replace(tzinfo=tz)
        if e_dt <= s_dt:
            raise ValueError("end_iso muss nach start_iso liegen")
    except Exception as e:
        raise ValueError(f"Ungültige Datumsangaben: {e}")
    # Serialisierung (immer Offset, kein naiver Wert)
    start_iso_norm = s_dt.isoformat()
    end_iso_norm = e_dt.isoformat()

    body = {
        "summary": summary,
        "start": {"dateTime": start_iso_norm, "timeZone": timezone},
        "end":   {"dateTime": end_iso_norm, "timeZone": timezone},
    }
    if description:
        body["description"] = description
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]
    if location:
        body["location"] = location
    if recurrence:
        # Erwartet vollständige RRULE Strings, z.B. "RRULE:FREQ=WEEKLY;COUNT=10"
        body["recurrence"] = recurrence
    if reminders_override_minutes or use_default_reminders is not None:
        rem: Dict[str, Any] = {}
        if use_default_reminders is not None:
            rem["useDefault"] = bool(use_default_reminders)
        if reminders_override_minutes:
            rem["overrides"] = [
                {"method": "popup", "minutes": int(m)} for m in reminders_override_minutes
            ]
            # Wenn overrides gesetzt, Standard nicht verwenden
            rem.setdefault("useDefault", False)
        if rem:
            body["reminders"] = rem

    service = _get_service()
    debug = os.getenv("CALENDAR_DEBUG", "0").lower() in {"1","true","yes","on"}
    cal_id = _resolve_calendar_id(service)
    if debug:
        print(f"[CAL][DEBUG] Insert -> cal={cal_id} {body['start']['dateTime']} - {body['end']['dateTime']} title='{summary}' timezone={timezone}")
    created = _with_retries(lambda: service.events().insert(calendarId=cal_id, body=body, sendUpdates="all").execute())
    # Optionale Verifikation (standardmäßig aktiv, kann via CALENDAR_SKIP_VERIFY=1 deaktiviert werden)
    if os.getenv("CALENDAR_SKIP_VERIFY", "0").lower() not in {"1","true","yes","on"}:
        try:
            fetched = _with_retries(lambda: service.events().get(calendarId=cal_id, eventId=created.get('id')).execute())
            if not fetched.get('id'):
                raise RuntimeError("Fetch nach Insert lieferte kein event id Feld – Event unsicher.")
        except Exception as ve:
            raise RuntimeError(f"Event-Erstellung unsicher – Fetch nach Insert fehlgeschlagen: {ve}")
    else:
        fetched = created
    if debug:
        print(f"[CAL][DEBUG] Created eventId={created.get('id')} link={created.get('htmlLink')}")
    result = {
        "eventId": created.get("id"),
        "htmlLink": created.get("htmlLink"),
        "summary": created.get("summary"),
        "start": created.get("start"),
        "end": created.get("end"),
        "verified": True,
        "calendarId": cal_id,
    }
    return result


def check_calendar_availability(start_iso: str,
                                end_iso: str,
                                timezone: str = "Europe/Berlin") -> Dict[str, Any]:
    """Prüft ob der Zeitraum frei ist (keine Überschneidung mit bestehenden Terminen).

    Nutzt die freebusy-API für genaue Belegung. Gibt eine strukturierte Antwort zurück:
      { "free": bool, "busy": [ {"start": ..., "end": ...}, ... ], "start": start_iso, "end": end_iso }

    Falls das Token zu geringe Berechtigungen hat (403), wird ein Hinweis zurückgegeben.
    """
    # Validierung & Parsing
    try:
        s_dt = datetime.fromisoformat(start_iso)
        e_dt = datetime.fromisoformat(end_iso)
        if e_dt <= s_dt:
            raise ValueError("end_iso muss nach start_iso liegen")
    except Exception as e:
        raise ValueError(f"Ungültige Datumsangaben: {e}")

    # Zeitzone anwenden falls naive
    try:
        if s_dt.tzinfo is None or e_dt.tzinfo is None:
            try:
                from zoneinfo import ZoneInfo  # Python 3.9+
                tz = ZoneInfo(timezone)
                if s_dt.tzinfo is None:
                    s_dt = s_dt.replace(tzinfo=tz)
                if e_dt.tzinfo is None:
                    e_dt = e_dt.replace(tzinfo=tz)
            except Exception:
                # Fallback: als UTC behandeln
                from datetime import timezone as _tz
                if s_dt.tzinfo is None:
                    s_dt = s_dt.replace(tzinfo=_tz.utc)
                if e_dt.tzinfo is None:
                    e_dt = e_dt.replace(tzinfo=_tz.utc)
    except Exception:
        pass

    service = _get_service()
    debug = os.getenv("CALENDAR_DEBUG", "0").lower() in {"1","true","yes","on"}
    cal_id = _resolve_calendar_id(service)
    body = {
        "timeMin": s_dt.isoformat(),
        "timeMax": e_dt.isoformat(),
        "items": [{"id": cal_id}],
    }
    try:
        if debug:
            print(f"[CAL][DEBUG] freebusy query {body['timeMin']} -> {body['timeMax']} cal={cal_id}")
        resp = _with_retries(lambda: service.freebusy().query(body=body).execute())
        cal = resp.get("calendars", {}).get(cal_id, {})
        busy = cal.get("busy", [])
        return {
            "start": s_dt.isoformat(),
            "end": e_dt.isoformat(),
            "timezone": timezone,
            "free": len(busy) == 0,
            "busy": busy,
        }
    except Exception as e:  # Fallback: versuche einfache Events-List Overlap
        try:
            events_resp = _with_retries(lambda: service.events().list(
                calendarId=cal_id,
                timeMin=(s_dt - timedelta(days=1)).isoformat(),  # etwas Puffer um Überlappungen zu finden
                timeMax=e_dt.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            ).execute())
            if debug:
                print("[CAL][DEBUG] Fallback events list used")
            items = events_resp.get('items', [])
            overlaps = []
            for ev in items:
                if ev.get('status') == 'cancelled':
                    continue
                ev_start_raw = ev.get('start', {}).get('dateTime') or ev.get('start', {}).get('date')
                ev_end_raw = ev.get('end', {}).get('dateTime') or ev.get('end', {}).get('date')
                try:
                    ev_s = datetime.fromisoformat(ev_start_raw.replace('Z', '+00:00'))
                    ev_e = datetime.fromisoformat(ev_end_raw.replace('Z', '+00:00'))
                except Exception:
                    continue
                # Überlappung wenn Start < gewünschtes Ende und Ende > gewünschtem Start
                if ev_s < e_dt and ev_e > s_dt:
                    overlaps.append({
                        'eventId': ev.get('id'),
                        'summary': ev.get('summary'),
                        'start': ev.get('start'),
                        'end': ev.get('end'),
                    })
            return {
                "start": s_dt.isoformat(),
                "end": e_dt.isoformat(),
                "timezone": timezone,
                "free": len(overlaps) == 0,
                "busy": overlaps,
                "warning": f"freebusy fehlgeschlagen: {e}. Fallback list verwendet." if overlaps else f"freebusy fehlgeschlagen: {e}",
            }
        except Exception as ee:
            return {
                "start": s_dt.isoformat(),
                "end": e_dt.isoformat(),
                "timezone": timezone,
                "free": False,
                "busy": [],
                "error": f"Verfügbarkeitsprüfung gescheitert: {ee}" ,
            }


def suggest_same_day_alternatives(start_iso: str,
                                  end_iso: str,
                                  timezone: str = "Europe/Berlin",
                                  max_suggestions: int = 3,
                                  workday_start: str = "08:00",
                                  workday_end: str = "18:00") -> Dict[str, Any]:
    """Schlägt freie Alternativ-Zeitfenster am selben Tag vor.

    Parameter:
      start_iso, end_iso: ursprünglicher (ggf. belegter) Wunschzeitraum
      timezone: IANA
      max_suggestions: Anzahl gewünschter Alternativen
      workday_start/workday_end: Grenzen des Arbeitstages (HH:MM)

    Rückgabe:
      { 'day': 'YYYY-MM-DD', 'original': {start,end}, 'duration_minutes': int,
        'suggestions': [ {start,end}, ... ], 'count': n }
    """
    try:
        s_dt = datetime.fromisoformat(start_iso)
        e_dt = datetime.fromisoformat(end_iso)
        if e_dt <= s_dt:
            raise ValueError("end_iso muss nach start_iso liegen")
    except Exception as e:
        raise ValueError(f"Ungültige Datumsangaben: {e}")

    # Zeitzonenanreicherung wie oben
    if s_dt.tzinfo is None or e_dt.tzinfo is None:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone)
            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=tz)
            if e_dt.tzinfo is None:
                e_dt = e_dt.replace(tzinfo=tz)
        except Exception:
            from datetime import timezone as _tz
            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=_tz.utc)
            if e_dt.tzinfo is None:
                e_dt = e_dt.replace(tzinfo=_tz.utc)

    day_str = s_dt.date().isoformat()
    wd_start_h, wd_start_m = map(int, workday_start.split(":"))
    wd_end_h, wd_end_m = map(int, workday_end.split(":"))
    day_start = s_dt.replace(hour=wd_start_h, minute=wd_start_m, second=0, microsecond=0)
    day_end = s_dt.replace(hour=wd_end_h, minute=wd_end_m, second=0, microsecond=0)
    duration = e_dt - s_dt

    # Liste der Events des Tages holen
    service = _get_service()
    cal_id = _resolve_calendar_id(service)
    events_resp = _with_retries(lambda: service.events().list(
        calendarId=cal_id,
        timeMin=day_start.isoformat(),
        timeMax=day_end.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute())
    items = [ev for ev in events_resp.get('items', []) if ev.get('status') != 'cancelled']

    busy_intervals = []
    for ev in items:
        ev_start_raw = ev.get('start', {}).get('dateTime') or ev.get('start', {}).get('date')
        ev_end_raw = ev.get('end', {}).get('dateTime') or ev.get('end', {}).get('date')
        if not ev_start_raw or not ev_end_raw:
            continue
        try:
            ev_s = datetime.fromisoformat(ev_start_raw.replace('Z', '+00:00'))
            ev_e = datetime.fromisoformat(ev_end_raw.replace('Z', '+00:00'))
        except Exception:
            continue
        busy_intervals.append((ev_s, ev_e))
    busy_intervals.sort(key=lambda x: x[0])

    # Merge überlappende busy intervals
    merged = []
    for interval in busy_intervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(list(interval))
        else:
            if interval[1] > merged[-1][1]:
                merged[-1][1] = interval[1]
    busy_intervals = [(a,b) for a,b in merged]

    # Freie Intervalle bestimmen
    free_intervals = []
    cursor = day_start
    for b_start, b_end in busy_intervals:
        if b_start > cursor:
            free_intervals.append((cursor, min(b_start, day_end)))
        if b_end > cursor:
            cursor = b_end
        if cursor >= day_end:
            break
    if cursor < day_end:
        free_intervals.append((cursor, day_end))

    suggestions: List[Dict[str, str]] = []
    # Wir wollen Slots ab ursprünglichem Start priorisieren; sortiere freie Intervalle entsprechend
    free_intervals.sort(key=lambda iv: (iv[0] < s_dt, iv[0]))  # Intervalle vor dem Wunschstart nach hinten

    for f_start, f_end in free_intervals:
        # Falls Intervall komplett vor Wunschstart liegt, optional überspringen
        # wir prüfen trotzdem ob noch Kapazität frei ist falls später nichts gefunden wird
        span = f_end - max(f_start, s_dt)
        candidate_start = max(f_start, s_dt)
        if span >= duration and candidate_start + duration <= f_end:
            suggestions.append({
                'start': candidate_start.isoformat(),
                'end': (candidate_start + duration).isoformat()
            })
        else:
            # evtl. passt der Slot früher am Tag
            span2 = f_end - f_start
            if span2 >= duration and f_start >= day_start and f_start >= s_dt:
                suggestions.append({
                    'start': f_start.isoformat(),
                    'end': (f_start + duration).isoformat()
                })
        if len(suggestions) >= max_suggestions:
            break

    return {
        'day': day_str,
        'timezone': timezone,
        'original': {'start': s_dt.isoformat(), 'end': e_dt.isoformat()},
        'duration_minutes': int(duration.total_seconds()//60),
        'suggestions': suggestions,
        'count': len(suggestions)
    }

def list_calendar_events(start_iso: str | None = None,
                         end_iso: str | None = None,
                         max_results: int = 5,
                         query: str | None = None,
                         include_cancelled: bool = False) -> Dict[str, Any]:
    """Listet kommende Kalender-Events.

    Parameter:
      start_iso: ISO Start (falls None -> jetzt)
      end_iso  : ISO Ende (falls None -> jetzt + 7 Tage)
      max_results: Anzahl der Events (1..50)
      query: Volltext-Suchstring (optional)
      include_cancelled: ob abgesagte Events einbezogen werden
    """
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    def _parse(ts: str | None, default):
        if not ts:
            return default
        try:
            # Unterstütze fehlendes 'Z' -> naive interpretieren als UTC
            dt = datetime.fromisoformat(ts.replace('Z','+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return default
    s_dt = _parse(start_iso, now)
    e_dt = _parse(end_iso, now + timedelta(days=7))
    if e_dt <= s_dt:
        e_dt = s_dt + timedelta(hours=1)
    time_min = s_dt.isoformat()
    time_max = e_dt.isoformat()
    service = _get_service()
    max_results = max(1, min(max_results, 50))
    cal_id = _resolve_calendar_id(service)
    events_result = _with_retries(lambda: service.events().list(
        calendarId=cal_id,
        timeMin=time_min,
        timeMax=time_max,
        maxResults=max_results,
        singleEvents=True,
        orderBy='startTime',
        q=query,
        showDeleted=include_cancelled,
    ).execute())
    items = events_result.get('items', [])
    simplified = []
    for ev in items:
        simplified.append({
            'eventId': ev.get('id'),
            'summary': ev.get('summary', '(ohne Titel)'),
            'start': ev.get('start'),
            'end': ev.get('end'),
            'status': ev.get('status'),
        })
    return {
        'count': len(simplified),
        'timeRange': {'start': time_min, 'end': time_max},
        'events': simplified,
    }

__all__ = ["create_calendar_event", "list_calendar_events", "check_calendar_availability", "suggest_same_day_alternatives"]
