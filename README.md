# Telefonassistent mit Google Cloud und Gemini

Dieses Projekt implementiert einen Telefonassistenten, der Sprache über ein virtuelles Audiokabel empfängt, transkribiert, mit Gemini verarbeitet und die Antwort wieder als Sprache ausgibt.

## Module
- **audio_io.py**: Audio-Ein-/Ausgabe
- **stt_google.py**: Speech-to-Text
- **gemini_api.py**: Gemini-Integration
- **tts_google.py**: Text-to-Speech
- **main.py**: Orchestrierung
- **config.py**: Konfiguration

## Kalender-Integration (Google Calendar)

Die Anwendung nutzt die Google Calendar API für:
1. Termin-Erstellung (`create_calendar_event`)
2. Verfügbarkeitsprüfung (`check_calendar_availability` über FreeBusy + Fallback)
3. Alternativ-Vorschläge am selben Tag (`suggest_same_day_alternatives`)
4. Auflisten von Terminen (`list_calendar_events`)

### Einrichtung
1. In der Google Cloud Console ein Projekt anlegen und die "Google Calendar API" aktivieren.
2. OAuth Client (Desktop / Installed App) erstellen – `client_secret.json` ins Projektverzeichnis legen (oder Pfad via `GOOGLE_OAUTH_CLIENT_SECRET`).
3. (Optional) Vorher vorhandenes `token.json` löschen, falls Scopes geändert wurden.
4. Abhängigkeiten installieren (s.u.). Beim ersten Start öffnet sich der Browser für den OAuth Flow.

### Installation Kalender-Abhängigkeiten
Die wichtigsten Pakete:
```powershell
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### Scopes
Standard: Vollzugriff (`https://www.googleapis.com/auth/calendar`).
Anpassbar über ENV `GOOGLE_CALENDAR_SCOPES` (Komma-separiert). Beispiele:
```powershell
$env:GOOGLE_CALENDAR_SCOPES='https://www.googleapis.com/auth/calendar.readonly'
```
Bei Wechsel auf readonly sind Schreiboperationen (Events anlegen) nicht mehr möglich.

### Dynamische Kalender-ID
Eine feste `CALENDAR_ID` ist optional. Verhalten:
- `CALENDAR_ID` leer / `primary` / `auto` => Primärer Kalender des authentifizierten Users
- Konkrete ID (z.B. geteilte Ressource) => genau dieser Kalender
Debug-Ausgabe (mit `CALENDAR_DEBUG=1`) zeigt aufgelöste ID: `AUTO resolved calendarId -> ...`

### Erweiterte Event-Felder
`create_calendar_event` akzeptiert zusätzliche optionale Parameter:
- `location`: Freitext-Ort
- `recurrence`: Liste von RRULE Strings (z.B. `RRULE:FREQ=WEEKLY;COUNT=10`)
- `reminders_override_minutes`: Liste Minutenwerte für Popup-Reminder
- `use_default_reminders`: True/False zur expliziten Steuerung

Beispiel Direktaufruf:
```python
from calendar_tools import create_calendar_event
create_calendar_event(
	summary="Projekt Sync",
	start_iso="2025-09-20T10:00:00",
	end_iso="2025-09-20T10:30:00",
	location="Videokonferenz",
	attendees=["team@example.com"],
	recurrence=["RRULE:FREQ=WEEKLY;COUNT=4"],
	reminders_override_minutes=[10,30]
)
```

### Verfügbarkeitsprüfung & Alternativen
- Primär: FreeBusy Endpoint
- Fallback: Events-Liste + manuelle Überlappungsprüfung
- Alternativen rechnen freie Slots innerhalb eines Arbeitstages (konfigurierbar in Funktion)

### Retry & Fehlertoleranz
Alle API-Aufrufe laufen über einen internen Retry mit exponentiellem Backoff (`_with_retries`) bei Statuscodes 429/5xx.

### Sicherheit / Best Practices
- Keine Secrets hardcoden – ENV Variablen + lokale `client_secret.json`.
- Token-Datei (`token.json`) nur lokal; bei Deployment verschlüsseln oder in Secret Store.
- Minimale Scopes setzen (z.B. `calendar.readonly` für reine Anzeige).
- Logs mit `CALENDAR_DEBUG=1` nur zeitweise aktivieren.

### Typische Environment Variablen
```powershell
$env:GEMINI_API_KEY='...'               # Gemini Key
$env:GOOGLE_OAUTH_CLIENT_SECRET='client_secret.json'
$env:GOOGLE_OAUTH_TOKEN='token.json'
$env:CALENDAR_ID='primary'              # oder explizite ID
$env:CALENDAR_DEBUG='1'                 # Debug Logging aktiv
$env:GOOGLE_CALENDAR_SCOPES='https://www.googleapis.com/auth/calendar'
$env:SPEECH_TRACE='1'                   # Erweiterte Sprach-/Entscheidungs-Traces (Variante A)
```

### Fehlerszenarien
| Problem | Ursache | Lösung |
|---------|---------|-------|
| 403 forbidden | Falscher Kalender / fehlender Zugriff | Andere ID testen / Freigabe prüfen |
| 404 not found | Kalender-ID falsch | ID korrigieren / `primary` nutzen |
| Keine Erstellung | Readonly Scope gesetzt | Scope anpassen + `token.json` löschen |
| Browser-Flow jedes Mal | Token kann nicht geschrieben werden | Schreibrechte im Verzeichnis prüfen |

### Mapping zur Checkliste
| Checkliste Punkt | Umsetzung |
|------------------|-----------|
| OAuth2 | `calendar_tools._load_credentials` mit Refresh + Reauth Flag |
| Events erstellen | `create_calendar_event` erweitert (Location, Recurrence, Reminders) |
| Events abrufen | `list_calendar_events` |
| Verfügbarkeit | `check_calendar_availability` + Fallback + Alternativen |
| Rate Limiting | `_with_retries` mit Backoff |
| Sicherheit | ENV Variablen, kein Hardcoding von Keys |
| Minimale Scopes | `GOOGLE_CALENDAR_SCOPES` konfigurierbar |
| Dokumentation | Dieser Abschnitt |

### Nächste mögliche Erweiterungen
- Unterstützung mehrerer paralleler Kalender (Auswahl per Sprachbefehl)
- Caching der FreeBusy Ergebnisse für kurze Zeitfenster
- UI (Streamlit) für visuelle Terminübersicht
- ICS Export / Mail Versand

### Diagnose Sprach-Assistent (Variante A)
Aktiviere detaillierte Entscheidungs- und Event-Trace-Ausgaben:
```powershell
$env:SPEECH_TRACE='1'
python speech_calendar_assistant.py
```
Beobachte Konsolenzeilen mit Präfix `[TRACE:SPEECH]` für:
- VAD Start/Ende, Segmentlängen
- Erkanntes STT Transkript & Parser Treffer
- Funktion Calls (Name + Parameter) und Event IDs
- Fehlerursachen bei direkter oder modellgestützter Erstellung

Deaktivieren durch `Remove-Item Env:SPEECH_TRACE` oder neues Terminal ohne Variable.

---

Für Voice-Bedienung siehe `speech_calendar_assistant.py`, für textbasierte Function Calls `gemini_function_chat.py`.


## Installation

```powershell
pip install -r requirements.txt
```

## Hinweise
- API-Keys in `config.py` eintragen
- Virtuelles Audiokabel konfigurieren und in `config.py` eintragen
