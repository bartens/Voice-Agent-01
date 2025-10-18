"""Sprachgesteuerter Kalender-Assistent

Funktion:
 - Wartet per VAD auf gesprochene Eingabe (z.B. "Lege morgen um 10 Uhr ein Meeting Projekt Sync für 30 Minuten mit alex@example.com an")
 - Nutzt Google STT -> Text
 - Sendet Text an Gemini mit Function Calling (create_calendar_event)
 - Führt bei Function Call das Event anlegen aus (calendar_tools.create_calendar_event)
 - Spielt Bestätigung per TTS ab

Stop-Wörter: stop, ende, abbrechen, quit

Voraussetzungen:
 - GEMINI_API_KEY gesetzt
 - Google Calendar OAuth (client_secret + token) vorhanden oder erster Flow
 - Mikrofon / Virtual Cable wie in config.py definiert
 - VAD aktiviert (config VAD_ENABLED True) für komfortables Einsprechen

Start:
   python speech_calendar_assistant.py

Hinweis: Das Modell kann Rückfragen stellen (fehlende Dauer, Datum etc.). Dann einfach Antwort sprechen.

Variante A (aktuell fokussiert):
    Ziel ist ausschließlich mehr Transparenz & Diagnostik (Instrumentation / Guard Rails),
    KEINE tiefgreifende Stream-Architekturänderung. Maßnahmen:
        - Zusätzliche Trace-/Entscheidungs-Logs für: VAD Segment (Länge, dBFS Peaks), STT Text, Parser Resultat,
            Wahl des Branches (Direkterstellung vs. Modell), Function Call Extraktion, Event-Erstellungsergebnis.
        - Aktivierung über neue ENV Variable `SPEECH_TRACE=1` (low overhead wenn nicht gesetzt).
    Nicht Bestandteil von Variante A (nur evtl. später): Adaptive Chunklängen, halbduplex Streaming,
    Zwischenhypothesen der STT, Interruptible TTS, dynamische Dauer-Fallbacks.
"""
from __future__ import annotations
import time
import threading
import os
import numpy as np
import sounddevice as sd
import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MIC_DEVICE_NAME,
    SPEAKER_DEVICE_NAME,
    VAD_ENABLED,
    VAD_START_DBFS,
    VAD_END_DBFS,
    VAD_END_HOLD,
    VAD_MIN_SECONDS,
    VAD_MAX_SECONDS,
    VAD_FRAME_MS,
    VAD_PREROLL_SECONDS,
)
import audio_io
from stt_google import transcribe_audio
from tts_google import synthesize_speech
from calendar_tools import create_calendar_event, check_calendar_availability, suggest_same_day_alternatives, pre_auth
from personality import GREETING_TEXT, SYSTEM_PROMPT_CALENDAR
from text_sanitize import sanitize_output
from date_utils import adjust_dates_if_year_missing
from session_context import init_session_time, get_session_date
import re
from datetime import datetime, timedelta, date

# --- SPEECH TRACE -----------------------------------------------------------
_SPEECH_TRACE_ENABLED = os.getenv("SPEECH_TRACE", "0") not in {"", "0", "false", "False", "FALSE"}

def speech_trace(*parts):
    """Leichtgewichtige Trace-Ausgabe (nur wenn SPEECH_TRACE=1 gesetzt)."""
    if _SPEECH_TRACE_ENABLED:
        try:
            msg = " ".join(str(p) for p in parts)
            print(f"[TRACE:SPEECH] {msg}")
        except Exception:
            pass

STOP_WORDS = {"stop", "ende", "abbrechen", "quit"}
TARGET_STT_SR = 16000

PRE_CREATION_ANNOUNCEMENT = "Vielen Dank. Ich werde prüfen ob der Termin frei ist. Das kann einen Moment dauern."

def _announce_pre_creation():
    """Spielt die kurze Vorab-Ansage bevor ein Termin tatsächlich erstellt/geprüft wird.

    Ziel: Dem Nutzer Feedback geben, dass jetzt Verfügbarkeits- / Insert-Operationen folgen.
    """
    try:
        _tts_play(PRE_CREATION_ANNOUNCEMENT)
    except Exception as e:
        speech_trace("Pre-creation announcement TTS error", e)

# Tool-Definition (wie gemini_function_chat, etwas DRY Duplikation für Isolation)
TOOLS = [{
    "function_declarations": [
        {
            "name": "check_calendar_availability",
            "description": "Prüft ob ein Zeitraum frei ist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {"type": "string", "description": "Start (YYYY-MM-DDTHH:MM:SS)."},
                    "end_iso": {"type": "string", "description": "Ende (YYYY-MM-DDTHH:MM:SS)."},
                    "timezone": {"type": "string", "description": "IANA Zeitzone"}
                },
                "required": ["start_iso", "end_iso"]
            }
        },
        {
            "name": "create_calendar_event",
            "description": "Erzeugt einen Google Kalender Termin (mit Ort, Wiederholung, Erinnerungen).",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Kurzer Titel."},
                    "start_iso": {"type": "string", "description": "Start (YYYY-MM-DDTHH:MM:SS)."},
                    "end_iso": {"type": "string", "description": "Ende (YYYY-MM-DDTHH:MM:SS)."},
                    "timezone": {"type": "string", "description": "IANA Zeitzone"},
                    "description": {"type": "string", "description": "Beschreibung / Details."},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "Emails"},
                    "location": {"type": "string", "description": "Ort oder Meeting-Link."},
                    "recurrence": {"type": "array", "items": {"type": "string"}, "description": "Liste RRULE Strings (z.B. 'RRULE:FREQ=DAILY;COUNT=5')."},
                    "reminders_override_minutes": {"type": "array", "items": {"type": "number"}, "description": "Popup-Erinnerungen in Minuten vor Start."},
                    "use_default_reminders": {"type": "boolean", "description": "Standard-Erinnerungen des Kalenders benutzen?"}
                },
                "required": ["summary", "start_iso", "end_iso"]
            }
        }
    ]
}]

SYSTEM_INSTRUCTION = SYSTEM_PROMPT_CALENDAR


def _pick_device(name_or_id, kind):
    devices = sd.query_devices()
    if isinstance(name_or_id, int):
        return name_or_id
    low = str(name_or_id).lower()
    for i,d in enumerate(devices):
        if d['name'].lower()==low and d[f'max_{kind}_channels']>0:
            return i
    for i,d in enumerate(devices):
        if low in d['name'].lower() and d[f'max_{kind}_channels']>0:
            return i
    di,do = sd.default.device
    return di if kind=='input' else do


def _downmix_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim==1:
        return arr.astype(np.int16)
    m = arr.mean(axis=1)
    return np.clip(m, -32768, 32767).astype(np.int16)


def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or sig.size==0:
        return sig
    x = sig.astype(np.float32)
    n_new = int(round(x.shape[0]*target_sr/orig_sr))
    if n_new < 2:
        return sig[:1]
    idx = np.linspace(0, x.shape[0]-1, n_new)
    out = np.interp(idx, np.arange(x.shape[0]), x)
    return np.clip(out, -32768, 32767).astype(np.int16)


def _peak_dbfs(samples: np.ndarray) -> float:
    if samples.size==0:
        return -120.0
    p = np.max(np.abs(samples))
    if p==0:
        return -120.0
    return 20*np.log10(p/32767.0)


def _record_vad_segment(mic_id: int, wait_timeout: float | None = None) -> np.ndarray:
    sr = audio_io.SAMPLE_RATE
    ch = audio_io.CHANNELS
    frame_len = int(sr * (VAD_FRAME_MS/1000.0))
    if frame_len < 160:
        frame_len = 160
    max_frames = int(VAD_MAX_SECONDS * sr / frame_len) + 2
    min_frames = int(VAD_MIN_SECONDS * sr / frame_len)
    silence_needed = int(VAD_END_HOLD * sr / frame_len)
    preroll_frames = int(VAD_PREROLL_SECONDS * sr / frame_len)
    collected = []
    preroll = []
    started=False
    silence_run = 0
    start_wait = time.time()
    stream = sd.InputStream(device=mic_id, samplerate=sr, channels=ch, dtype='int16', blocksize=frame_len)
    with stream:
        for idx in range(max_frames):
            block,_ = stream.read(frame_len)
            mono = _downmix_mono(block)
            mono16 = _resample(mono, sr, TARGET_STT_SR)
            peak = _peak_dbfs(mono16)
            if _SPEECH_TRACE_ENABLED and idx % 10 == 0:
                speech_trace(f"VAD frame={idx} peak={peak:.1f}dB started={started}")
            if not started:
                preroll.append(block.copy())
                if len(preroll) > preroll_frames:
                    preroll.pop(0)
                if peak >= VAD_START_DBFS:
                    started=True
                    collected.extend(preroll)
                    collected.append(block.copy())
                    speech_trace("VAD START detected", f"preroll_frames={len(preroll)}")
                else:
                    if wait_timeout and (time.time()-start_wait) >= wait_timeout:
                        return np.zeros((0,), dtype=np.int16)
                    if (time.time()-start_wait) >= VAD_MAX_SECONDS:
                        return np.zeros((0,), dtype=np.int16)
            else:
                collected.append(block.copy())
                if peak < VAD_END_DBFS:
                    silence_run +=1
                else:
                    silence_run = 0
                dur = len(collected)*frame_len/sr
                if _SPEECH_TRACE_ENABLED and idx % 25 == 0:
                    speech_trace(f"VAD dur={dur:.2f}s silence_run={silence_run}")
                if dur >= VAD_MAX_SECONDS:
                    speech_trace("VAD forced stop: max duration reached")
                    break
                if dur >= VAD_MIN_SECONDS and silence_run >= silence_needed:
                    speech_trace("VAD END detected via silence")
                    break
    if not collected:
        speech_trace("VAD result: empty segment")
        return np.zeros((0,), dtype=np.int16)
    total = np.concatenate(collected, axis=0)
    speech_trace("VAD result length samples=", total.shape[0])
    return total


def _tts_play(text: str):
    audio = synthesize_speech(text, sample_rate=TARGET_STT_SR)
    play = _resample(audio, TARGET_STT_SR, audio_io.SAMPLE_RATE)
    if audio_io.CHANNELS>1:
        play = np.repeat(play[:,None], audio_io.CHANNELS, axis=1)
    audio_io.play_audio_chunk(play)


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
    """Sichere Extraktion von Text ohne direkten resp.text Zugriff."""
    try:
        texts = []
        for cand in getattr(resp, 'candidates', []) or []:
            content = getattr(cand, 'content', None)
            if not content:
                continue
            for part in getattr(content, 'parts', []) or []:
                t = getattr(part, 'text', None)
                if t:
                    t = t.strip()
                    if t:
                        texts.append(t)
        if texts:
            return "\n".join(texts)
        raw = getattr(resp, 'text', None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    except Exception:
        pass
    try:
        frs = [str(getattr(cand, 'finish_reason', '')) for cand in getattr(resp, 'candidates', []) if getattr(cand, 'finish_reason', None) is not None]
        if frs:
            return f"(keine Antwort – finish_reason={','.join(frs)})"
    except Exception:
        pass
    return "(keine Antwort)"


# --- Natürliche deutsche Datums-/Zeit-Parser Hilfen ---
_WEEKDAY_MAP = {
    'montag': 0, 'dienstag': 1, 'mittwoch': 2, 'donnerstag': 3,
    'freitag': 4, 'samstag': 5, 'sonntag': 6,
}

def _parse_relative_date(token_str: str, today: date | None = None) -> date | None:
    """Parst einfache relative Angaben: heute, morgen, übermorgen, in X tagen, in X wochen, nächste woche, übernächste woche, wochentag (nächster)."""
    if today is None:
        today = date.today()
    s = token_str.lower().strip()
    if s in {"heute"}:
        return today
    if s in {"morgen"}:
        return today + timedelta(days=1)
    if s in {"übermorgen", "uebermorgen"}:
        return today + timedelta(days=2)
    if s in {"nächste woche", "naechste woche"}:
        return today + timedelta(days=7 - today.weekday())  # nächster Wochenmontag
    if s in {"übernächste woche", "uebernaechste woche"}:
        return today + timedelta(days=7 - today.weekday() + 7)
    number_words = {
        'eins':1,'ein':1,'eine':1,'einen':1,'zwei':2,'drei':3,'vier':4,'fuenf':5,'fünf':5,
        'sechs':6,'sieben':7,'acht':8,'neun':9,'zehn':10,'elf':11,'zwoelf':12,'zwölf':12
    }
    m = re.match(r"in\s+(\d+)\s+tagen?", s)
    if m:
        return today + timedelta(days=int(m.group(1)))
    m = re.match(r"in\s+(\d+)\s+wochen?", s)
    if m:
        return today + timedelta(weeks=int(m.group(1)))
    m = re.match(r"in\s+([a-zäöüß]+)\s+tagen?", s)
    if m and m.group(1) in number_words:
        return today + timedelta(days=number_words[m.group(1)])
    m = re.match(r"in\s+([a-zäöüß]+)\s+wochen?", s)
    if m and m.group(1) in number_words:
        return today + timedelta(weeks=number_words[m.group(1)])
    # Wochentag (nächster vorkommender)
    if s in _WEEKDAY_MAP:
        target = _WEEKDAY_MAP[s]
        delta = (target - today.weekday()) % 7
        if delta == 0:
            delta = 7  # nächster gleicher Wochentag -> eine Woche später
        return today + timedelta(days=delta)
    return None

def _parse_time_fragment(text: str) -> tuple[int,int] | None:
    """Extrahiert HH:MM oder HHMM oder HH aus einem Textfragment."""
    t = text.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", t)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{3,4})$", t)
    if m:
        raw = m.group(1).zfill(4)
        return int(raw[:2]), int(raw[2:])
    m = re.match(r"^(\d{1,2})$", t)
    if m:
        return int(m.group(1)), 0
    return None

def _preparse_natural_german(utterance: str) -> dict | None:
    """Versucht aus einer Alltagsphrase strukturierte Termininfos zu extrahieren.

    Erkannt werden:
      - relative Datumsangaben (morgen, übermorgen, in 3 tagen, nächste woche, montag, ...)
      - absolute deutsche Daten dd.mm.yyyy oder dd.mm.
      - Zeit: "um 10", "um 10:30", "10:30", "1030"
      - Dauer: "für 30 minuten", "30 minuten", "30 min"
      - Teilnehmer: email Muster
      - Titel: "titel: ..." oder Schlüsselwort nach "meeting", "termin", "besprechung" wenn gefolgt von Freitext
    Rückgabe: dict mit keys summary,start_iso,end_iso,attendees
    """
    txt = utterance.strip()
    low = txt.lower()
    # Nutze fixiertes Session-Datum (initialisiert beim Start)
    today = get_session_date()
    # Email sammeln
    emails = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", txt)
    # Dauer
    dur = None
    m = re.search(r"(\d{1,3})\s*(?:minuten|min)\b", low)
    if m:
        dur = int(m.group(1))
    # Datum (absolut)
    date_obj = None
    m = re.search(r"(\d{1,2})[.](\d{1,2})[.](\d{4})", low)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            date_obj = date(y, mo, d)
        except Exception:
            pass
    else:
        # ohne Jahr -> dieses oder nächstes falls schon vorbei
        m = re.search(r"(\d{1,2})[.](\d{1,2})[.]", low)
        if m:
            d, mo = map(int, m.groups())
            y = today.year
            try:
                cand = date(y, mo, d)
                if cand < today:
                    cand = date(y+1, mo, d)
                date_obj = cand
            except Exception:
                pass
    # Relative Tokens
    if not date_obj:
        rel_candidates = ["heute","morgen","übermorgen","uebermorgen","nächste woche","naechste woche","übernächste woche","uebernaechste woche"] + list(_WEEKDAY_MAP.keys())
        for rc in rel_candidates:
            if rc in low:
                parsed = _parse_relative_date(rc, today)
                if parsed:
                    date_obj = parsed
                    break
        if not date_obj:
            # Numerische relative Angaben
            m = re.search(r"in\s+\d+\s+(?:tagen?|wochen?)", low)
            if m:
                parsed = _parse_relative_date(m.group(0), today)
                if parsed:
                    date_obj = parsed
            # Zahlwort relative Angaben (eins, zwei ...)
            if not date_obj:
                m = re.search(r"in\s+[a-zäöüß]+\s+(?:tagen?|wochen?)", low)
                if m:
                    parsed = _parse_relative_date(m.group(0), today)
                    if parsed:
                        date_obj = parsed
    # Zeit
    hour = minute = None
    m = re.search(r"um\s+(\d{1,2}[:.]?\d{0,2})", low)
    if m:
        frag = m.group(1)
        tm = _parse_time_fragment(frag.replace('.', ':'))
        if tm:
            hour, minute = tm
    if hour is None:
        # Fallback: erstes Zeitfragment
        m = re.search(r"\b(\d{1,2}:\d{2}|\d{3,4})\b", low)
        if m:
            tm = _parse_time_fragment(m.group(1))
            if tm:
                hour, minute = tm
    # Titel
    title = None
    m = re.search(r"titel\s*:\s*([^,;]+)", low)
    if m:
        title = m.group(1).strip().title()
    if not title:
        # heuristik: mentiones of meeting/termin/besprechung + folgendes Wort(e)
        m = re.search(r"(meeting|termin|besprechung)\s+([\wäöüß\- ]{3,40})", low)
        if m:
            title = (m.group(1) + " " + m.group(2)).strip().title()
    if not title:
        title = "Termin"
    # Wenn wir Datum, Zeit und Dauer haben -> strukturieren
    if date_obj and hour is not None and dur:
        try:
            start_dt = datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute or 0, 0)
            end_dt = start_dt + timedelta(minutes=dur)
            return {
                'summary': title,
                'start_iso': start_dt.isoformat(),
                'end_iso': end_dt.isoformat(),
                'attendees': emails or None,
                'parsed': True,
            }
        except Exception:
            return None
    return None


def run(stop_event: threading.Event | None = None):
    if not GEMINI_API_KEY:
        print("[FEHLER] GEMINI_API_KEY fehlt")
        return
    if not VAD_ENABLED:
        print("[HINWEIS] VAD_ENABLED False – aktiviere für komfortable Nutzung.")
    genai.configure(api_key=GEMINI_API_KEY)
    # Session Zeit initialisieren (nur einmal pro Prozess)
    session_dt = init_session_time()
    dynamic_system_instruction = SYSTEM_INSTRUCTION + f"\nHeutiges Datum (Europe/Berlin): {session_dt.date().isoformat()}."
    model = genai.GenerativeModel(GEMINI_MODEL_NAME, tools=TOOLS, system_instruction=dynamic_system_instruction)
    chat = model.start_chat()
    mic_id = _pick_device(MIC_DEVICE_NAME, 'input')
    spk_id = _pick_device(SPEAKER_DEVICE_NAME, 'output')
    # Pre-Auth vor jeglicher Audioausgabe erzwingen
    print("[INFO] Prüfe Kalender OAuth...")
    if not pre_auth():
        print("[FEHLER] Kalender-OAuth nicht erfolgreich. Bitte client_secret.json prüfen oder Token löschen.")
        return
    print(GREETING_TEXT)
    speech_trace("Session start", f"session_dt={session_dt.isoformat()}")
    # Geräte-Info nicht mehr ausgeben (Anforderung)
    # Begrüßung zuerst per TTS ausgeben (nach erfolgreichem OAuth)
    try:
        _tts_play(GREETING_TEXT)
    except Exception as e:
        print("[WARN] TTS-Begrüßung fehlgeschlagen:", e)
    history_turn = 0
    # Auto-Stop nach konfigurierbarer Stillenzeit
    AUTO_STOP_SILENCE_SECONDS = float(os.getenv("ASSISTANT_AUTO_STOP_SILENCE", "10"))
    SILENCE_SLICE = 1.0  # Sekunden pro VAD-Warteintervall
    accumulated_silence = 0.0

    try:
        while True:
            if stop_event and stop_event.is_set():
                print("[INFO] Externer Stopp angefordert – beende Assistent.")
                break
            history_turn +=1
            print("\n[WARTEN] Sprich jetzt...")
            seg = _record_vad_segment(mic_id, wait_timeout=SILENCE_SLICE)
            speech_trace("Loop turn", history_turn, "raw_segment_samples=", seg.shape[0])
            if seg.size==0:
                accumulated_silence += SILENCE_SLICE
                if accumulated_silence >= AUTO_STOP_SILENCE_SECONDS:
                    print(f"[INFO] Automatischer Stopp nach {AUTO_STOP_SILENCE_SECONDS:.0f}s Stille.")
                    speech_trace("Auto-stop due to silence", accumulated_silence)
                    break
                # Kurzer Hinweis nur sparsam, um Spam zu vermeiden
                # print("(Stille)")
                continue
            accumulated_silence = 0.0
            mono = _downmix_mono(seg)
            mono16 = _resample(mono, audio_io.SAMPLE_RATE, TARGET_STT_SR)
            try:
                text = transcribe_audio(mono16, sample_rate=TARGET_STT_SR)
            except Exception as e:
                print("[ERR] STT:", e)
                speech_trace("STT error", e)
                continue
            if not text:
                print("(leer)")
                speech_trace("Empty STT result")
                continue
            print("User:", text)
            speech_trace("User text=", text)
            if text.lower().strip() in STOP_WORDS:
                print("Beende.")
                speech_trace("Stop word detected – terminating")
                break
            # Vorparser für natürliche deutsche Datums-/Zeitangaben
            parsed = _preparse_natural_german(text)
            if parsed:
                speech_trace("Parser hit", parsed)
            else:
                speech_trace("Parser miss – fallback to model")
            if parsed and parsed.get('parsed'):
                try:
                    # Anwenden von adjust_dates_if_year_missing gewährleisten (auch wenn Jahr implizit war)
                    ns, ne, adj = adjust_dates_if_year_missing(text, parsed['start_iso'], parsed['end_iso'])
                    if adj:
                        parsed['start_iso'] = ns
                        parsed['end_iso'] = ne
                        speech_trace("Year adjusted", ns, ne)
                    _announce_pre_creation()
                    result = create_calendar_event(
                        summary=parsed['summary'],
                        start_iso=parsed['start_iso'],
                        end_iso=parsed['end_iso'],
                        timezone='Europe/Berlin',
                        attendees=parsed.get('attendees')
                    )
                    speech_trace("Direct create OK", result.get('id'))
                    confirm = f"Erstellt: {result.get('summary')} {result['start']['dateTime']}"
                    print("Assistent:", confirm)
                    _tts_play(confirm)
                    continue
                except Exception as e:
                    print("[Parser/Direct] Fehler direkte Erstellung:", e)
                    speech_trace("Direct create error", e)
                    # Weiter zum Modell-Fallback
            else:
                # Dokumentiere warum kein direkter Parser-Pfad genutzt wurde
                if parsed is None:
                    speech_trace("Parser returned None -> model path")
                elif not parsed.get('parsed'):
                    speech_trace("Parser dict ohne parsed=True -> model path", parsed)
            resp = chat.send_message(text)
            speech_trace("Model primary response recv")
            fc = _extract_function_call(resp)
            if not fc:
                reply = sanitize_output(_safe_text(resp) or "(leer)")
                print("Assistent:", reply)
                try:
                    _tts_play(reply)
                except Exception:
                    pass
                speech_trace("No function call", reply[:120] if reply else None)
                continue
            name, args = fc
            speech_trace("Function call extracted", name, args)
            if name == 'check_calendar_availability':
                if 'timezone' not in args or not args.get('timezone'):
                    args['timezone'] = 'Europe/Berlin'
                try:
                    result = check_calendar_availability(**args)
                    speech_trace("Availability result", result)
                    _send_function_response(chat, name, result)
                    if result.get('free'):
                        # Zeitraum frei – Modell darf ggf. direkt create_calendar_event aufrufen
                        follow = chat.send_message("Antworte gemäß System-Prompt auf Basis der Function-Response (free=true).")
                        speech_trace("Follow after free=true received")
                    else:
                        # Zeitraum belegt – Alternativvorschläge berechnen lassen
                        alt_args = {
                            'start_iso': args['start_iso'],
                            'end_iso': args['end_iso'],
                            'timezone': args['timezone']
                        }
                        try:
                            alt_res = suggest_same_day_alternatives(**alt_args)
                            _send_function_response(chat, 'suggest_same_day_alternatives', alt_res)
                            follow = chat.send_message("Antworte gemäß System-Prompt: Zeitraum belegt, biete Alternativen an und frage nach Auswahl oder neuer Angabe.")
                            speech_trace("Alternatives provided", alt_res)
                        except Exception as e:
                            follow = chat.send_message(f"Zeitraum belegt. Konnte Alternativen nicht berechnen ({e}). Bitte frage nach anderer Zeit am selben Tag.")
                            speech_trace("Alternative calc error", e)
                    # Prüfen, ob das Follow bereits einen weiteren Function Call (z.B. create_calendar_event) enthält
                    fc2 = _extract_function_call(follow)
                    if fc2:
                        name2, args2 = fc2
                        speech_trace("Second function call", name2, args2)
                        if name2 == 'create_calendar_event':
                            if 'timezone' not in args2 or not args2.get('timezone'):
                                args2['timezone'] = 'Europe/Berlin'
                            try:
                                if 'start_iso' in args2 and 'end_iso' in args2:
                                    ns, ne, adj = adjust_dates_if_year_missing(str(args2), args2['start_iso'], args2['end_iso'])
                                    if adj:
                                        args2['start_iso'] = ns
                                        args2['end_iso'] = ne
                                        chat.send_message(f"Hinweis: Jahr ergänzt -> {ns} bis {ne}.")
                                        speech_trace("Year adjusted 2nd", ns, ne)
                                _announce_pre_creation()
                                result2 = create_calendar_event(**args2)
                                _send_function_response(chat, name2, result2)
                                follow2 = chat.send_message("Bestätige sehr knapp (Titel + Start).")
                                confirm = sanitize_output(_safe_text(follow2) or "Termin angelegt.")
                                print("Bestätigung:", confirm)
                                _tts_play(confirm)
                                speech_trace("Second create OK", result2.get('id'))
                                # Log wichtige Felder komprimiert
                                speech_trace("Second create summary", result2.get('summary'), result2.get('start',{}).get('dateTime'))
                            except Exception as e2:
                                err2 = f"Fehler: {e2}"
                                print(err2)
                                _send_function_response(chat, name2, {"error": err2})
                                try:
                                    _tts_play("Fehler beim Anlegen. Bitte wiederholen.")
                                except Exception:
                                    pass
                                speech_trace("Second create error", e2)
                                speech_trace("Second create error args", {k: v for k,v in args2.items() if k in ('summary','start_iso','end_iso')})
                        else:
                            # Unerwarteter Function Call: normal behandeln
                            reply = sanitize_output(_safe_text(follow) or "(keine Antwort)")
                            _tts_play(reply)
                            speech_trace("Unexpected function call type", name2)
                    else:
                        reply = sanitize_output(_safe_text(follow) or "(keine Antwort)")
                        print("Assistent:", reply)
                        _tts_play(reply)
                        speech_trace("No second function call", reply[:120] if reply else None)
                except Exception as e:
                    err = f"Fehler Verfügbarkeit: {e}"
                    print(err)
                    _send_function_response(chat, name, {"error": err})
                    try:
                        _tts_play("Fehler bei der Verfügbarkeitsprüfung.")
                    except Exception:
                        pass
                    speech_trace("Availability error", e)
                continue
            elif name == 'create_calendar_event':  # unterstützt jetzt location, recurrence, reminders_override_minutes, use_default_reminders
                if 'timezone' not in args or not args.get('timezone'):
                    args['timezone'] = 'Europe/Berlin'
                try:
                    # Automatische Jahreskorrektur falls User kein Jahr nannte und Termin sonst in der Vergangenheit läge
                    if 'start_iso' in args and 'end_iso' in args:
                        ns, ne, adj = adjust_dates_if_year_missing(text, args['start_iso'], args['end_iso'])
                        if adj:
                            args['start_iso'] = ns
                            args['end_iso'] = ne
                            chat.send_message(f"Hinweis: Jahr ergänzt -> {ns} bis {ne}.")
                            speech_trace("Year adjusted direct model", ns, ne)
                    _announce_pre_creation()
                    result = create_calendar_event(**args)
                    _send_function_response(chat, name, result)
                    follow = chat.send_message("Bestätige kurz.")
                    confirm = sanitize_output(_safe_text(follow) or "Termin angelegt.")
                    print("Bestätigung:", confirm)
                    _tts_play(confirm)
                    speech_trace("Model create OK", result.get('id'))
                    speech_trace("Model create summary", result.get('summary'), result.get('start',{}).get('dateTime'))
                except Exception as e:
                    err = f"Fehler: {e}"
                    print(err)
                    _send_function_response(chat, name, {"error": err})
                    try:
                        _tts_play("Fehler beim Anlegen. Bitte wiederholen.")
                    except Exception:
                        pass
                    speech_trace("Model create error", e)
                    speech_trace("Model create error args", {k: v for k,v in args.items() if k in ('summary','start_iso','end_iso')})
                continue
                msg = "Funktion nicht unterstützt."
                print(msg)
                _tts_play(msg)
                continue
    except KeyboardInterrupt:
        print("\nAbbruch (KeyboardInterrupt)")
    print("Fertig.")


def main():
    run()


if __name__ == '__main__':  # pragma: no cover
    main()