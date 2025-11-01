"""Chat mit Gemini + Function Calling für Google Kalender und Kontakte.

Ablauf:
 - User Prompt -> Modell
 - Modell kann normalen Text oder function_call (Kalender/Kontakte) liefern
 - Bei function_call: lokale Ausführung -> function_response zurück -> Modell bestätigt

Start: python gemini_function_chat.py
"""
from __future__ import annotations
import os
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from personality import SYSTEM_PROMPT_CALENDAR, SYSTEM_PROMPT_GENERAL, SYSTEM_PROMPT_CONTACTS
from calendar_tools import create_calendar_event, list_calendar_events, check_calendar_availability, suggest_same_day_alternatives
from contacts import add_contact, list_contacts, get_contact, delete_contact
from text_sanitize import sanitize_output
from date_utils import adjust_dates_if_year_missing
import re
from datetime import datetime, timedelta

_MODEL: genai.GenerativeModel | None = None
_CHAT = None

def _ensure_model():
    """Lazy Initialisierung für stillen Import ohne Seiteneffekte.

    Initialisiert SDK Konfiguration, Modell und Chat erst bei erster Nutzung.
    """
    global _MODEL, _CHAT
    if _MODEL is not None and _CHAT is not None:
        return
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY nicht gesetzt.")
    genai.configure(api_key=GEMINI_API_KEY)
    _MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME, tools=TOOLS, system_instruction=SYSTEM_INSTRUCTION)
    _CHAT = _MODEL.start_chat()
    return

TOOLS = [{
    "function_declarations": [
        {
            "name": "create_calendar_event",
            "description": "Erzeugt einen Google Kalender Termin (unterstützt Ort, Wiederholung und Erinnerungen).",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Kurzer Titel des Termins."},
                    "start_iso": {"type": "string", "description": "Startzeit ISO 8601 (YYYY-MM-DDTHH:MM:SS)."},
                    "end_iso": {"type": "string", "description": "Endzeit ISO 8601 (YYYY-MM-DDTHH:MM:SS)."},
                    "timezone": {"type": "string", "description": "IANA Zeitzone (Standard Europe/Berlin)"},
                    "description": {"type": "string", "description": "Beschreibung / Details."},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "Email Liste."},
                    "location": {"type": "string", "description": "Ort oder Meeting-Link."},
                    "recurrence": {"type": "array", "items": {"type": "string"}, "description": "Liste von RRULE Strings (z.B. 'RRULE:FREQ=WEEKLY;COUNT=5')."},
                    "reminders_override_minutes": {"type": "array", "items": {"type": "number"}, "description": "Popup-Erinnerungen in Minuten vor Beginn (setzt use_default_reminders automatisch false falls gesetzt)."},
                    "use_default_reminders": {"type": "boolean", "description": "Standard-Erinnerungen des Kalenders verwenden?"}
                },
                "required": ["summary", "start_iso", "end_iso"]
            }
        },
        {
            "name": "list_calendar_events",
            "description": "Listet kommende Kalender-Events in einem Zeitraum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {"type": "string", "description": "Start des Zeitfensters (ISO). Optional."},
                    "end_iso": {"type": "string", "description": "Ende des Zeitfensters (ISO). Optional."},
                    "max_results": {"type": "number", "description": "Anzahl der zurückzugebenden Events (1-50)."},
                    "query": {"type": "string", "description": "Freitext-Suche (optional)."},
                    "include_cancelled": {"type": "boolean", "description": "Abgesagte Events einschließen?"}
                },
                "required": []
            }
        },
        {
            "name": "check_calendar_availability",
            "description": "Prüft ob der angegebene Zeitraum frei ist (keine Überschneidung).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {"type": "string", "description": "Startzeit ISO 8601."},
                    "end_iso": {"type": "string", "description": "Endzeit ISO 8601."},
                    "timezone": {"type": "string", "description": "IANA Zeitzone (Standard Europe/Berlin)"}
                },
                "required": ["start_iso", "end_iso"]
            }
        },
        {
            "name": "suggest_same_day_alternatives",
            "description": "Schlägt freie Alternativen am selben Tag für den angegebenen Zeitraum vor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {"type": "string", "description": "Ursprünglicher Start ISO"},
                    "end_iso": {"type": "string", "description": "Ursprüngliches Ende ISO"},
                    "timezone": {"type": "string"},
                    "max_suggestions": {"type": "number", "description": "Max Anzahl Vorschläge (Standard 3)"}
                },
                "required": ["start_iso", "end_iso"]
            }
        },
        {
            "name": "add_contact",
            "description": "Fügt einen neuen Kundenkontakt hinzu oder aktualisiert einen bestehenden.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name des Kunden (erforderlich)."},
                    "phone": {"type": "string", "description": "Telefonnummer."},
                    "email": {"type": "string", "description": "E-Mail-Adresse."},
                    "address": {"type": "string", "description": "Adresse."},
                    "notes": {"type": "string", "description": "Notizen zum Kunden."}
                },
                "required": ["name"]
            }
        },
        {
            "name": "list_contacts",
            "description": "Listet alle Kundenkontakte alphabetisch sortiert auf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {"type": "string", "description": "Optionaler Suchbegriff zum Filtern nach Name."}
                },
                "required": []
            }
        },
        {
            "name": "get_contact",
            "description": "Ruft die Details eines bestimmten Kundenkontakts ab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name des Kunden."}
                },
                "required": ["name"]
            }
        },
        {
            "name": "delete_contact",
            "description": "Löscht einen Kundenkontakt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name des zu löschenden Kunden."}
                },
                "required": ["name"]
            }
        }
    ]
}]

# Kombiniere allgemeine Persona mit dem kalender- und kontakt-spezifischen Tool-Prompt
SYSTEM_INSTRUCTION = SYSTEM_PROMPT_GENERAL + "\n\n" + SYSTEM_PROMPT_CALENDAR + "\n\n" + SYSTEM_PROMPT_CONTACTS

# Modell/Chat werden nun erst bei erstem handle_user_message()-Aufruf erstellt.


def _extract_function_call(resp) -> tuple[str, dict] | None:
    try:
        for cand in resp.candidates:
            for part in cand.content.parts:
                fc = getattr(part, 'function_call', None)
                if fc:
                    return fc.name, dict(fc.args)
    except Exception:
        return None
    return None


def _send_function_response(chat, name: str, result: dict):
    """Sendet Function-Result korrekt (kein zusätzliches Listen-Wrapping)."""
    chat.send_message({
        "role": "function",
        "parts": [{
            "function_response": {
                "name": name,
                "response": result,
            }
        }]
    })


def _safe_text(resp) -> str:
    """Robuste Extraktion von Modelltext ohne Nutzung resp.text Quick-Accessor.

    Falls keine Text-Parts vorhanden (z.B. finish_reason=STOP ohne Content oder Safety-Block), wird ein
    Platzhalter zurückgegeben.
    """
    try:
        texts = []
        for cand in getattr(resp, 'candidates', []) or []:
            for part in getattr(cand, 'content', {}).parts:
                t = getattr(part, 'text', None)
                if t:
                    texts.append(t)
        if texts:
            # Kombiniere Mehrfachteile defensiv
            return "\n".join([t.strip() for t in texts if t.strip()]) or "(keine Antwort)"
        # Fallback: Versuche generisches Attribut
        raw = getattr(resp, 'text', None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    except Exception:
        pass
    # Letzte Eskalation: Finish Reason anzeigen für Debug
    try:
        frs = [getattr(cand, 'finish_reason', None) for cand in getattr(resp, 'candidates', [])]
        frs = [str(fr) for fr in frs if fr is not None]
        if frs:
            return f"(keine Antwort vom Modell – finish_reason={','.join(frs)})"
    except Exception:
        pass
    return "(keine Antwort)"


def handle_user_message(text: str) -> str:
    """Einfacher Single-Round Function-Calling Ablauf.

    - Sendet die Nutzereingabe an das Modell.
    - Falls ein function_call zurückkommt: führt genau diese Funktion aus, sendet eine function_response und
      holt danach EIN Bestätigungs- / Zusammenfassungs-Reply vom Modell.
    - Kein automatisches Nach-Tool-Chaining, keine Fallback-Auto-Erstellung.
    """
    if not text or not text.strip():
        return "(leer – keine Eingabe)"

    # Vorparser: deutscher Befehl wie
    # "erstelle einen termin mit anna am 17.09.2025 um 1000 für 30 minuten, titel: meeting, email: anna@email.de"
    # Unterstützt Varianten: "um 10:00" / "um 1000" / Dauer: "für 30 minuten" / Titel optional / Email optional / mit <Name>
    low = text.lower()
    pattern = re.compile(r"erstelle\s+einen?\s+termin(?:\s+mit\s+(?P<person>[^,]+?))?\s+am\s+(?P<date>\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})\s+um\s+(?P<time>\d{1,2}[:.]?\d{2})\s+für\s+(?P<dur>\d{1,3})\s+min", re.IGNORECASE)
    m = pattern.search(low)
    if m:
        try:
            raw_date = m.group('date')
            raw_time = m.group('time')
            dur_min = int(m.group('dur'))
            # Normalisiere Datum
            # Erlaube Trennzeichen .-/
            dparts = re.split(r"[.\-/]", raw_date)
            day, month, year = map(int, dparts)
            # Zeit normalisieren
            t = raw_time.replace('.', ':')
            if ':' not in t:
                # Form HHMM
                if len(t) in {3,4}:  # z.B. 900 oder 1000
                    t = t.zfill(4)
                    t = t[:2] + ':' + t[2:]
                elif len(t) == 2:
                    t = t + ':00'
            if t.count(':') == 1:
                t = t + ':00'
            dt_start = datetime(year, month, day, int(t[:2]), int(t[3:5]), int(t[6:8]))
            dt_end = dt_start + timedelta(minutes=dur_min)
            # Titel extrahieren
            title_match = re.search(r"titel\s*:\s*([^,]+)", low)
            summary = title_match.group(1).strip() if title_match else "Termin"
            email_match = re.search(r"email\s*:\s*([^,\s]+)", low)
            attendees = [email_match.group(1).strip()] if email_match else []
            # Direkte Erstellung (überspringe Modell, um Robustheit sicherzustellen)
            result = create_calendar_event(
                summary=summary,
                start_iso=dt_start.isoformat(),
                end_iso=dt_end.isoformat(),
                timezone='Europe/Berlin',
                attendees=attendees or None,
                description=f"Automatisch erstellt aus Eingabe: {text}"
            )
            return sanitize_output(f"Erstellt: {result.get('summary')} {result['start']['dateTime']} -> {result['end']['dateTime']}")
        except Exception as pe:
            # Falls Vorparser scheitert, normaler Modellweg
            pass
    _ensure_model()
    resp = _CHAT.send_message(text)
    fc = _extract_function_call(resp)
    if not fc:
        return sanitize_output(_safe_text(resp) or "(leer)")
    name, args = fc
    if os.getenv("GEMINI_FUNC_DEBUG", "0").lower() in {"1","true","yes","on"}:
        print(f"[FUNC][DEBUG] {name} args={args}")
    try:
        if name == "create_calendar_event":  # unterstützt nun auch location, recurrence, reminders_override_minutes, use_default_reminders
            if 'timezone' not in args or not args.get('timezone'):
                args['timezone'] = 'Europe/Berlin'
            orig_start, orig_end = args['start_iso'], args['end_iso']
            new_start, new_end, adjusted = adjust_dates_if_year_missing(text, orig_start, orig_end)
            if adjusted:
                args['start_iso'] = new_start
                args['end_iso'] = new_end
            result = create_calendar_event(**args)
        elif name == "list_calendar_events":
            result = list_calendar_events(**args)
        elif name == "check_calendar_availability":
            if 'timezone' not in args or not args.get('timezone'):
                args['timezone'] = 'Europe/Berlin'
            result = check_calendar_availability(**args)
        elif name == "suggest_same_day_alternatives":
            if 'timezone' not in args or not args.get('timezone'):
                args['timezone'] = 'Europe/Berlin'
            result = suggest_same_day_alternatives(**args)
        elif name == "add_contact":
            result = add_contact(**args)
        elif name == "list_contacts":
            result = list_contacts(**args)
        elif name == "get_contact":
            result = get_contact(**args)
        elif name == "delete_contact":
            result = delete_contact(**args)
        else:
            _send_function_response(_CHAT, name, {"error": f"Unbekannte Funktion {name}"})
            return f"Nicht unterstützte Funktion: {name}"
    except RuntimeError as e:
        msg = ("Kalender nicht authentifiziert oder Credentials Problem. "
               f"Detail: {e}")
        _send_function_response(_CHAT, name, {"error": msg})
        return msg
    except FileNotFoundError as e:
        msg = ("Erforderliche Datei fehlt (client_secret.json). "
               f"Detail: {e}")
        _send_function_response(_CHAT, name, {"error": msg})
        return msg
    except Exception as e:
        err = f"Fehler bei {name}: {e}"
        _send_function_response(_CHAT, name, {"error": err})
        return err

    _send_function_response(_CHAT, name, result)
    follow = _CHAT.send_message("Bestätige oder fasse kurz zusammen.")
    return sanitize_output(_safe_text(follow) or "(keine Antwort)")


def main():
    print("Kalender-Chat bereit. 'exit' zum Beenden.")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue  # keine leere Anfrage an Modell senden (verhindert ValueError)
        answer = handle_user_message(user)
        print("Bot:", answer)

if __name__ == "__main__":  # pragma: no cover
    main()
