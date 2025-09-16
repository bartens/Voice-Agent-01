"""Log Trigger Listener

Überwacht eine (MicroSIP) Logdatei und startet den Sprachassistenten automatisch,
 sobald ein definierter Trigger-Logeintrag auftaucht (z.B. eingehender/angenommener Anruf).

Vorgehen:
 1. Pfad zur Log-Datei konfigurieren (ENV oder unten LOG_PATH).
 2. REGEX_PATTERNS definieren für Einträge, die den Start auslösen.
 3. Skript starten; es tailt die Datei (polling, kein fs watcher nötig) und startet bei Match den Assistenten.
 4. Während der Assistent läuft werden neue Trigger ignoriert (Debounce/Cooldown).

Beispiel-Trigger für MicroSIP (abhängig von Einstellungen):
   "Call answered" / "incoming call" / SIP 200 OK / INVITE
Passe die Muster an deine tatsächliche Log-Ausgabe an.
"""
from __future__ import annotations
import os
import re
import time
import threading
from typing import List

from speech_calendar_assistant import run as run_calendar_assistant
import threading as _th

"""Konfigurationsparameter (teils via ENV überschreibbar)

CALL_LOG_PATH          Pfad zur Logdatei
CALL_LOG_PATTERNS      Semikolon-getrennte Regex Muster (ersetzen Default)
CALL_LOG_COOLDOWN_S    Cooldown Sekunden
CALL_LOG_TAIL_INTERVAL Poll-Intervall Sekunden
CALL_LOG_ONE_SHOT      true -> Beende nach erstem Trigger
CALL_LOG_PRINT_MATCH   true/false – Match-Zeile ausgeben
"""

LOG_PATH = os.getenv("CALL_LOG_PATH", r"C:\\Users\\ebart\\AppData\\Local\\MicroSIP\\microsip_log.txt")

def _compile_patterns() -> List[re.Pattern]:
    env_pat = os.getenv("CALL_LOG_PATTERNS")
    pats: List[str]
    if env_pat:
        pats = [p.strip() for p in env_pat.split(";") if p.strip()]
    else:
        pats = [
            r"incoming call",
            r"call answered",
            r"\b200 OK\b",
        ]
    compiled: List[re.Pattern] = []
    for p in pats:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            print(f"[LOGTRIGGER] Ungültiges Regex ignoriert: {p} ({e})")
    return compiled

REGEX_PATTERNS: List[re.Pattern] = _compile_patterns()

# Zusätzliche Default-Startmuster (falls nicht via ENV überschrieben) – nur ergänzende Info
DEFAULT_ENHANCED_START = [
    r"Start talksprut..",
    r"Jitter buffer starts returning normal frames",
    r"state changed to CONFIRMED",
    r"Response msg 200/INVITE",
]

# End-Trigger Muster (konfigurierbar)
def _compile_end_patterns() -> List[re.Pattern]:
    env_end = os.getenv("CALL_LOG_END_PATTERNS")
    pats: List[str]
    if env_end:
        pats = [p.strip() for p in env_end.split(";") if p.strip()]
    else:
        pats = [
            r"Request msg BYE",
            r"Call \d+: deinitializing media",
            r"Closing sound device after idle",
        ]
    out: List[re.Pattern] = []
    for p in pats:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            print(f"[LOGTRIGGER] Ungültiges End-Regex ignoriert: {p} ({e})")
    return out

END_PATTERNS = _compile_end_patterns()

def _get_float(env_name: str, default: float) -> float:
    try:
        return float(os.getenv(env_name, str(default)))
    except Exception:
        return default

COOLDOWN_SECONDS = _get_float("CALL_LOG_COOLDOWN_S", 5.0)
TAIL_INTERVAL = _get_float("CALL_LOG_TAIL_INTERVAL", 0.4)
PRINT_MATCH_LINES = os.getenv("CALL_LOG_PRINT_MATCH", "true").lower() in {"1","true","on","yes"}
ONE_SHOT = os.getenv("CALL_LOG_ONE_SHOT", "false").lower() in {"1","true","on","yes"}

_running_assistant = threading.Event()
_stop_flag = threading.Event()
_last_end = 0.0
ASSISTANT_STOP_EVENT = threading.Event()
_assistant_thread: _th.Thread | None = None


def _assistant_wrapper():
    global _last_end
    try:
        run_calendar_assistant(stop_event=ASSISTANT_STOP_EVENT)
    except Exception as e:
        print(f"[LOGTRIGGER] Assistent Fehler: {e}")
    finally:
        _running_assistant.clear()
        _last_end = time.time()
        ASSISTANT_STOP_EVENT.clear()


def _line_is_trigger(line: str) -> bool:
    # Negative Filter: Registrierung ignorieren, um REGISTER 200 OK nicht als Anrufstart zu werten
    if 'register' in line.lower():
        return False
    for rx in REGEX_PATTERNS:
        if rx.search(line):
            return True
    return False


def _line_is_end(line: str) -> bool:
    for rx in END_PATTERNS:
        if rx.search(line):
            return True
    return False


def tail_log_and_trigger():
    print(f"[LOGTRIGGER] Datei: {LOG_PATH}")
    print(f"[LOGTRIGGER] Start-Muster: {[r.pattern for r in REGEX_PATTERNS]}")
    print(f"[LOGTRIGGER] End-Muster: {[r.pattern for r in END_PATTERNS]}")
    if ONE_SHOT:
        print("[LOGTRIGGER] ONE_SHOT aktiv – beendet nach erstem Trigger.")
    # Warten bis Datei existiert
    while not os.path.isfile(LOG_PATH) and not _stop_flag.is_set():
        print("[LOGTRIGGER] Warte auf Log-Datei...")
        time.sleep(1.5)
    if _stop_flag.is_set():
        return
    pos = 0
    last_size = 0
    f = None
    try:
        f = open(LOG_PATH, 'r', encoding='utf-8', errors='ignore')
        f.seek(0, os.SEEK_END)  # nur neue Zeilen
        pos = f.tell()
        last_size = pos
        while not _stop_flag.is_set():
            try:
                # Stat prüfen (Rotation / Truncation)
                try:
                    st = os.stat(LOG_PATH)
                    if st.st_size < pos:
                        # verkleinert -> neu öffnen
                        print("[LOGTRIGGER] Truncation/Rotation erkannt – Neustart am Anfang")
                        f.close()
                        f = open(LOG_PATH, 'r', encoding='utf-8', errors='ignore')
                        pos = 0
                        last_size = 0
                except FileNotFoundError:
                    # Datei gerade rotiert – warten bis wieder da
                    time.sleep(1.0)
                    continue
                f.seek(pos)
                line = f.readline()
                if not line:
                    time.sleep(TAIL_INTERVAL)
                    continue
                pos = f.tell()
                lstr = line.strip()
                if not lstr:
                    continue
                if _line_is_trigger(lstr):
                    if PRINT_MATCH_LINES:
                        print(f"[LOGTRIGGER] Match: {lstr}")
                    if _running_assistant.is_set():
                        continue
                    if (time.time() - _last_end) < COOLDOWN_SECONDS:
                        continue
                    print("[LOGTRIGGER] Starte Assistent...")
                    _running_assistant.set()
                    ASSISTANT_STOP_EVENT.clear()
                    global _assistant_thread
                    _assistant_thread = threading.Thread(target=_assistant_wrapper, daemon=False, name="AssistantThread")
                    _assistant_thread.start()
                    if ONE_SHOT:
                        break
                elif _line_is_end(lstr):
                    if PRINT_MATCH_LINES:
                        print(f"[LOGTRIGGER] (END) {lstr}")
                    # Externen Stopp anfordern, falls Assistent läuft
                    if _running_assistant.is_set():
                        print("[LOGTRIGGER] Sende Stop-Event an Assistent...")
                        ASSISTANT_STOP_EVENT.set()
                        # Optional begrenztes Warten auf Threadende
                        if _assistant_thread and _assistant_thread.is_alive():
                            _assistant_thread.join(timeout=10)
            except Exception as e:
                print(f"[LOGTRIGGER] Fehler beim Lesen: {e}")
                time.sleep(1.0)
    finally:
        if f:
            try:
                f.close()
            except Exception:
                pass
    print("[LOGTRIGGER] Beendet.")


def main():
    try:
        tail_log_and_trigger()
    except KeyboardInterrupt:
        print("\n[LOGTRIGGER] Abbruch")
        _stop_flag.set()


if __name__ == '__main__':  # pragma: no cover
    main()
