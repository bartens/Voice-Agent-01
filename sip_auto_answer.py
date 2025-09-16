"""SIP Auto-Answer + Übergabe an Sprachassistent (Variante 3 Basis).

Hinweis: Dies ist ein Startpunkt mit pjsua (pjsip). Du benötigst:
  pip install pjsua
Je nach Plattform evtl. vorkompilierte Wheels oder Build Toolchain.

Ablauf:
 1. SIP Account registrieren
 2. Auf eingehende INVITE warten
 3. Automatisch 200 OK (wenn SIP_AUTO_ANSWER)
 4. Audiostream empfangen -> in PCM Frames wandeln
 5. Übergabe an bestehende VAD/Transkriptions-Pipeline (hier Platzhalter)
 6. Antworten via TTS zurück in Outbound RTP streamen

Dieser Prototyp integriert noch NICHT direkt die vorhandene VAD/LLM Loop –
Stubs sind markiert (# TODO integrate) damit incremental erweitert werden kann.

Sicherheit/Datenschutz: Vor produktivem Einsatz rechtliche Ansagen beachten.
"""
from __future__ import annotations
import threading
import time
import queue
import sys
import os
import wave
import tempfile
import numpy as np

# Versuche zuerst pjsua2 (neuere Python-Bindings), dann pjsua. Beide erfordern lokal gebaute PJSIP libs.
pj = None  # wird auf Modul gesetzt wenn Import klappt
_pj_import_error = None
for _mod_name in ("pjsua2", "pjsua"):
    try:
        pj = __import__(_mod_name)  # type: ignore
        break
    except ImportError as _e:  # pragma: no cover - nur Laufzeitdiagnose
        _pj_import_error = _e
        pj = None
if pj is None:
    # Kurzer Hinweis für Laufzeit: detaillierte Build-Anleitung im Fehlertext
    print("[SIP] Warnung: Keine PJSIP Python-Bindings gefunden (pjsua2 oder pjsua). "
          "Bitte PJSIP bauen: VS Build Tools + pjproject + 'python setup.py build/install' im Ordner pjsip-apps/src/python.")

from config import (
    SIP_REGISTRAR,
    SIP_USERNAME,
    SIP_PASSWORD,
    SIP_DOMAIN,
    SIP_LOCAL_PORT,
    SIP_AUTO_ANSWER,
    SIP_GREETING_ENABLED,
    SIP_GREETING_TEXT,
    VAD_START_DBFS,
    VAD_END_DBFS,
    VAD_END_HOLD,
    VAD_MIN_SECONDS,
    VAD_MAX_SECONDS,
)

from stt_google import transcribe_audio
from tts_google import synthesize_speech
from text_sanitize import sanitize_output
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from personality import SYSTEM_PROMPT_GENERAL, SYSTEM_PROMPT_CALENDAR
from calendar_tools import create_calendar_event, list_calendar_events, check_calendar_availability, suggest_same_day_alternatives

# Repliziere Tool-Definition (ähnlich gemini_function_chat) für direkte Einbettung
TOOLS = [{
    "function_declarations": [
        {"name": "create_calendar_event", "description": "Erzeugt Termin.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "start_iso": {"type": "string"}, "end_iso": {"type": "string"}, "timezone": {"type": "string"}, "description": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["summary", "start_iso", "end_iso"]}},
        {"name": "list_calendar_events", "description": "Listet Events.", "parameters": {"type": "object", "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}, "max_results": {"type": "number"}, "query": {"type": "string"}, "include_cancelled": {"type": "boolean"}}, "required": []}},
        {"name": "check_calendar_availability", "description": "Prüft Verfügbarkeit.", "parameters": {"type": "object", "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}, "timezone": {"type": "string"}}, "required": ["start_iso", "end_iso"]}},
        {"name": "suggest_same_day_alternatives", "description": "Alternativen gleicher Tag.", "parameters": {"type": "object", "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}, "timezone": {"type": "string"}}, "required": ["start_iso", "end_iso"]}}
    ]
}]

SYSTEM_INSTRUCTION = SYSTEM_PROMPT_GENERAL + "\n\n" + SYSTEM_PROMPT_CALENDAR

chat_model = None
chat_session = None

def _init_chat():
    global chat_model, chat_session
    if not GEMINI_API_KEY:
        log("Warnung: Kein GEMINI_API_KEY – Antworten werden nur transkribiert zurückgegeben.")
        return
    if chat_model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        chat_model = genai.GenerativeModel(GEMINI_MODEL_NAME, tools=TOOLS, system_instruction=SYSTEM_INSTRUCTION)
        chat_session = chat_model.start_chat()

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

def _send_function_response(name: str, result: dict):
    if not chat_session:
        return
    chat_session.send_message({
        "role": "function",
        "parts": [{"function_response": {"name": name, "response": result}}]
    })

def _handle_llm(text: str) -> str:
    if not chat_session:
        return "Ich habe dich verstanden: " + text
    resp = chat_session.send_message(text)
    fc = _extract_function_call(resp)
    if not fc:
        return sanitize_output(resp.text or "(leer)")
    name, args = fc
    try:
        if name == 'create_calendar_event':
            if 'timezone' not in args or not args.get('timezone'):
                args['timezone'] = 'Europe/Berlin'
            result = create_calendar_event(**args)
            _send_function_response(name, result)
            follow = chat_session.send_message("Bestätige kurz.")
            return sanitize_output(follow.text or "(keine Bestätigung)")
        elif name == 'list_calendar_events':
            result = list_calendar_events(**args)
            _send_function_response(name, result)
            follow = chat_session.send_message("Fasse kurz zusammen.")
            return sanitize_output(follow.text or "(keine Zusammenfassung)")
        elif name == 'check_calendar_availability':
            if 'timezone' not in args or not args.get('timezone'):
                args['timezone'] = 'Europe/Berlin'
            result = check_calendar_availability(**args)
            _send_function_response(name, result)
            follow = chat_session.send_message("Sag frei oder belegt und nächsten Schritt.")
            return sanitize_output(follow.text or "(keine Antwort)")
        elif name == 'suggest_same_day_alternatives':
            if 'timezone' not in args or not args.get('timezone'):
                args['timezone'] = 'Europe/Berlin'
            result = suggest_same_day_alternatives(**args)
            _send_function_response(name, result)
            follow = chat_session.send_message("Erkläre Alternativen und frage ob eine passt.")
            return sanitize_output(follow.text or "(keine Antwort)")
    except Exception as e:
        _send_function_response(name, {"error": str(e)})
        return f"Fehler bei Funktion {name}: {e}"
    return "Unbekannte Funktion."

# ---- Simple Logging Wrapper ----

def log(msg: str):
    print(f"[SIP] {msg}")

# ---- Media Handling Placeholders ----

class AudioBridge(pj.AudioMedia):  # type: ignore
    """Empfängt Audio von PJSIP, stellt Rohdaten via Callback bereit.

    Vereinfachter Ansatz: Wir nutzen getPortInfo() nicht; pjsua liefert
    uns Media über die Konferenzbrücke. Für komplexere Verarbeitung
    wäre ein eigener Media Port (pj.Lib.create_media_port) nötig.
    """
    pass

# Puffer für eingehende Samples (Placeholder – Integration mit STT Pipeline)
INCOMING_PCM = queue.Queue(maxsize=50)  # Rohdaten int16

def _peak_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    p = np.max(np.abs(samples))
    if p == 0:
        return -120.0
    return 20 * np.log10(p / 32767.0)

class Segmentor(threading.Thread):
    """Einfacher VAD basierend auf Peak dBFS wie im bestehenden Assistenten.
    Liest kontinuierlich Frames aus INCOMING_PCM und bildet Segmente.
    """
    def __init__(self, call_handle_provider, sample_rate=16000, frame_ms=30):
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000.0)
        self.running = True
        self.call_handle_provider = call_handle_provider
        self.silence_frames_needed = int(VAD_END_HOLD / (frame_ms/1000.0))

    def run(self):
        _init_chat()
        buf = np.zeros(0, dtype=np.int16)
        collecting = False
        collected = []
        silence_run = 0
        start_time = None
        max_frames_total = int(VAD_MAX_SECONDS * 1000 / 30)
        frames_in_segment = 0
        while self.running:
            try:
                frame = INCOMING_PCM.get(timeout=0.2)
            except queue.Empty:
                continue
            buf = np.concatenate([buf, frame])
            while buf.size >= self.frame_len:
                cur = buf[:self.frame_len]
                buf = buf[self.frame_len:]
                peak = _peak_dbfs(cur)
                if not collecting:
                    if peak >= VAD_START_DBFS:
                        collecting = True
                        collected = [cur]
                        frames_in_segment = 1
                        silence_run = 0
                        start_time = time.time()
                else:
                    collected.append(cur)
                    frames_in_segment += 1
                    if peak < VAD_END_DBFS:
                        silence_run += 1
                    else:
                        silence_run = 0
                    dur = frames_in_segment * self.frame_len / self.sample_rate
                    if dur >= VAD_MAX_SECONDS or (dur >= VAD_MIN_SECONDS and silence_run >= self.silence_frames_needed):
                        # Segment fertig
                        segment = np.concatenate(collected)
                        threading.Thread(target=self._process_segment, args=(segment,), daemon=True).start()
                        collecting = False
                        collected = []
                        silence_run = 0
                        frames_in_segment = 0

    def _process_segment(self, segment: np.ndarray):
        if segment.size == 0:
            return
        try:
            text = transcribe_audio(segment, sample_rate=16000)
        except Exception as e:
            log(f"STT Fehler: {e}")
            return
        if not text:
            return
        log(f"USER: {text}")
        reply = _handle_llm(text)
        log(f"BOT: {reply}")
        self._play_tts(reply)

    def _play_tts(self, text: str):
        if not text:
            return
        try:
            audio = synthesize_speech(text, sample_rate=16000)
        except Exception as e:
            log(f"TTS Fehler: {e}")
            return
        # Schreibe WAV und spiele via player
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(tmp_fd)
        try:
            with wave.open(tmp_path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(audio.tobytes())
            lib = pj.Lib.instance()
            player_id = lib.create_player(tmp_path, loop=0)
            player_slot = lib.player_get_slot(player_id)
            call = self.call_handle_provider()
            if call:
                lib.conf_connect(player_slot, call.info().conf_slot)
            play_duration = audio.shape[0] / 16000.0 + 0.2
            threading.Thread(target=self._cleanup_player, args=(player_id, tmp_path, play_duration), daemon=True).start()
        except Exception as e:
            log(f"Playback Fehler: {e}")

    def _cleanup_player(self, player_id, path, delay):
        time.sleep(delay)
        try:
            lib = pj.Lib.instance()
            lib.player_destroy(player_id)
        except Exception:
            pass
        try:
            os.remove(path)
        except Exception:
            pass

# ---- Call Callback ----

class MyCallCallback(pj.CallCallback):  # type: ignore
    def __init__(self, call):
        super().__init__(call)
        self.rec_id = None
        self.tail_thread = None

    def on_state(self):
        ci = self.call.info()
        log(f"Call-State: {ci.state_text} ({ci.last_reason})")
        if ci.state == pj.CallState.DISCONNECTED:
            log("Call beendet")
        elif ci.state == pj.CallState.CONFIRMED:
            log("Call verbunden")
            if SIP_GREETING_ENABLED:
                log(f"Begrüßung: {SIP_GREETING_TEXT}")

    def on_media_state(self):
        ci = self.call.info()
        if ci.media_state == pj.MediaState.ACTIVE:
            log("Media aktiv – Audio verbunden")
            try:
                call_med = self.call.info().conf_slot
                lib = pj.Lib.instance()
                tmp_wav = os.path.join(tempfile.gettempdir(), f"sip_in_{int(time.time())}.wav")
                self.rec_id = lib.create_recorder(tmp_wav)
                rec_slot = lib.recorder_get_slot(self.rec_id)
                lib.conf_connect(call_med, rec_slot)
                log(f"Recorder gestartet: {tmp_wav}")
                self.tail_thread = threading.Thread(target=self._tail_wav, args=(tmp_wav,), daemon=True)
                self.tail_thread.start()
            except Exception as e:
                log(f"Media Connect Fehler: {e}")

    def on_dtmf_digit(self, digits):  # type: ignore
        log(f"DTMF empfangen: {digits}")

    def _tail_wav(self, path: str):
        waited = 0
        while waited < 5 and (not os.path.exists(path) or os.path.getsize(path) < 44):
            time.sleep(0.1)
            waited += 0.1
        if not os.path.exists(path):
            return
        pos = 44
        sample_width = 2
        channels = 1
        while True:
            try:
                if self.call.info().state == pj.CallState.DISCONNECTED:  # type: ignore
                    break
            except Exception:
                break
            try:
                size = os.path.getsize(path)
            except Exception:
                break
            if size > pos:
                to_read = size - pos
                with open(path, 'rb') as f:
                    f.seek(pos)
                    data = f.read(to_read)
                pos += to_read
                if len(data) % (sample_width*channels) != 0:
                    pos -= (len(data) % (sample_width*channels))
                    continue
                arr = np.frombuffer(data, dtype=np.int16)
                try:
                    INCOMING_PCM.put(arr, timeout=0.1)
                except queue.Full:
                    pass
            else:
                time.sleep(0.05)

# ---- Account Callback ----

class MyAccountCallback(pj.AccountCallback):  # type: ignore
    def __init__(self, account=None):
        super().__init__(account)
        self.service_ref = None

    def on_incoming_call(self, call):  # type: ignore
        log("Eingehender Anruf")
        call_cb = MyCallCallback(call)
        call.set_callback(call_cb)
        # Referenz im Service aktualisieren
        if self.service_ref:
            self.service_ref.current_call = call
        if SIP_AUTO_ANSWER:
            log("Auto-Answer aktiv – sende 200 OK")
            call.answer(200)
        else:
            log("Auto-Answer aus – Klingeln lassen (180)")
            call.answer(180)

# ---- SIP Hauptklasse ----

class SipService:
    def __init__(self):
        if not pj:
            raise RuntimeError("pjsua Modul nicht verfügbar. Bitte 'pip install pjsua' versuchen.")
        self.lib = pj.Lib()
        self.acc = None
        self.running = False
        self.segmentor = None
        self.current_call = None

    def start(self):
        log("Initialisiere PJSUA...")
        ua_cfg = pj.UAConfig()
        ua_cfg.user_agent = "AssistantSIP/0.1"
        media_cfg = pj.MediaConfig()
        media_cfg.clock_rate = 16000  # passend zu STT Pipeline
        media_cfg.snd_clock_rate = 16000
        media_cfg.channel_count = 1
        media_cfg.ec_tail_len = 0  # Echo Cancelling ggf. aktivieren
        self.lib.init(ua_cfg=ua_cfg, media_cfg=media_cfg, log_cfg=pj.LogConfig(level=3, callback=lambda l, m: None))

        # Transport (UDP)
        port = SIP_LOCAL_PORT if SIP_LOCAL_PORT > 0 else 0
        self.lib.create_transport(pj.TransportType.UDP, pj.TransportConfig(port=port))
        self.lib.start()
        log("PJSUA gestartet")

        if not SIP_REGISTRAR or not SIP_USERNAME or not SIP_PASSWORD:
            log("SIP Zugangsdaten unvollständig – Registrierung übersprungen")
            return

        acc_cfg = pj.AccountConfig(domain=SIP_DOMAIN, username=SIP_USERNAME, password=SIP_PASSWORD)
        acc_cfg.id = f"sip:{SIP_USERNAME}@{SIP_DOMAIN}"
        acc_cfg.reg_uri = f"sip:{SIP_REGISTRAR}"
        acc_cfg.allow_contact_rewrite = True
        self.acc = self.lib.create_account(acc_cfg)
        acc_cb = MyAccountCallback(self.acc)
        acc_cb.service_ref = self
        self.acc.set_callback(acc_cb)
        log("Registrierung initiiert")

        # Segmentor starten (liest globalen Queue)
        self.segmentor = Segmentor(lambda: self.current_call)
        self.segmentor.start()

    def loop(self):
        self.running = True
        try:
            while self.running:
                # Aktualisiere aktuelle aktive Call Referenz (einfachster Fall: erster aktiver)
                if self.acc:
                    for c in self.acc.info().calls:
                        # pjsua Python hat nicht immer acc.info().calls; fallback via lib
                        pass
                time.sleep(0.2)
        except KeyboardInterrupt:
            log("Stop durch KeyboardInterrupt")
        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False
        try:
            if self.acc:
                self.acc.delete()
            self.lib.destroy()
            self.lib = None
        except Exception:
            pass
        log("Beendet")


def main():
    svc = SipService()
    try:
        svc.start()
        svc.loop()
    except Exception as e:
        log(f"Fehler: {e}")
        try:
            svc.stop()
        except Exception:
            pass

if __name__ == "__main__":  # pragma: no cover
    main()
