"""Multi-Turn Sprach-Konversation

Ziel: Mehrere aufeinanderfolgende Runden (User spricht -> STT -> Gemini Antwort -> TTS Playback)
mit einfachem Gesprächskontext.

Funktionsweise (vereinfachte Turn-Erkennung):
 - Jede Runde: feste Aufnahmedauer (--seconds) oder vorzeitiger Abbruch bei sehr leiser Energie.
 - STT -> Text; falls leer: Runde wird verworfen (max. --max-empty hintereinander bevor Abbruch).
 - Gesprächsverlauf wird als Liste gespeichert und bei jeder Anfrage an Gemini als komprimierter Kontext
   (letzte --context-turns Runden) erneut gesendet.
 - Stopp-Wort: Wenn Nutzer 'stop', 'ende', 'quit' oder 'tschüss' sagt, wird beendet.

CLI Optionen:
  --seconds <float>        Aufnahmedauer pro Turn (Default 4.0)
  --max-turns <int>        Maximale Anzahl Konversationsrunden (Default 10)
  --context-turns <int>    Wie viele letzte Runden in den Prompt übernehmen (Default 6)
  --silence-dbfs <float>   Schwellwert (Peak dBFS) unter dem ein Turn als Stille verworfen wird (Default -45)
  --max-empty <int>        Maximale aufeinanderfolgende leere / stille Versuche (Default 3)
  --raw                    Roh-Transkript bei Leere anzeigen

Abbruch jederzeit mit CTRL+C.

Hinweis: Dies ist kein Streaming / Echtzeit Code – für Echtzeit müssten wir auf Streaming STT umstellen.
"""
"""Multi-turn conversation demo (legacy, round-based) with optional VAD.

Hinweis:
    Alle VAD Parameter stammen jetzt ausschließlich aus config.py (VAD_* Konstanten).
    Die früheren CLI-Flags für VAD existieren nicht mehr.

Empfehlung:
    Für reine kontinuierliche VAD-Konversation ohne feste Runden
    nutze das Skript multi_turn_vad_only.py.

Dieses Skript hier behält die Rundenzahl und (falls VAD deaktiviert) eine feste
Aufnahmedauer pro Turn bei, dient also eher zu Vergleichs-/Testzwecken.
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import sounddevice as sd
import time

from config import (
    MIC_DEVICE_NAME,
    SPEAKER_DEVICE_NAME,
    GEMINI_API_KEY,
    VAD_ENABLED,
    VAD_START_DBFS,
    VAD_END_DBFS,
    VAD_END_HOLD,
    VAD_MIN_SECONDS,
    VAD_MAX_SECONDS,
    VAD_FRAME_MS,
    VAD_PREROLL_SECONDS,
)
import audio_io  # für SAMPLE_RATE, CHANNELS, play_audio_chunk
from stt_google import transcribe_audio
from gemini_api import get_gemini_response
from tts_google import synthesize_speech

TARGET_STT_SR = 16000
STOP_WORDS = {"stop", "ende", "quit", "tschüss", "tschuss"}


def _pick_device(requested: str | int, kind: str) -> int:
    devices = sd.query_devices()
    if isinstance(requested, int):
        if 0 <= requested < len(devices) and devices[requested][f'max_{kind}_channels']>0:
            return requested
        raise RuntimeError(f"Ungültige Geräte-ID {requested} für {kind}")
    req_lower = str(requested).lower()
    exact = [i for i,d in enumerate(devices) if d['name'].lower()==req_lower and d[f'max_{kind}_channels']>0]
    if exact: return exact[0]
    subs = [i for i,d in enumerate(devices) if req_lower in d['name'].lower() and d[f'max_{kind}_channels']>0]
    if subs: return subs[0]
    # fallback
    di, do = sd.default.device
    return di if kind=='input' else do

def _downmix_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim==1: return data.astype(np.int16)
    mono = data.mean(axis=1)
    return np.clip(mono, -32768, 32767).astype(np.int16)


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or data.size==0: return data
    x = data.astype(np.float32)
    n_new = int(round(x.shape[0]*target_sr/orig_sr))
    if n_new<2: return data[:1]
    idx_src = np.linspace(0, x.shape[0]-1, n_new)
    res = np.interp(idx_src, np.arange(x.shape[0]), x)
    return np.clip(res, -32768, 32767).astype(np.int16)


def _analyze_peak_dbfs(samples: np.ndarray) -> float:
    if samples.size==0: return -120.0
    peak = np.max(np.abs(samples))
    if peak==0: return -120.0
    return 20*np.log10(peak/32767.0)


def _record_vad_turn(mic_id: int):
    """Nimmt Audio mit einfacher Energie-VAD auf.

    Ablauf:
      1. Frames (Blockgrößen) von vad_frame_ms werden nacheinander aufgezeichnet.
      2. Startbedingung: Peak dBFS >= vad_start_dbfs (ansonsten warten bis max_seconds erreicht -> leere Rückgabe)
      3. Wenn gestartet: Stille-Frames (Peak < vad_end_dbfs) zählen; wenn deren kumulative Dauer >= vad_end_hold und min_seconds erfüllt -> Stopp
      4. Harte Abbrüche: > vad_max_seconds
    """
    sr = audio_io.SAMPLE_RATE
    ch = audio_io.CHANNELS
    frame_len = int(sr * (VAD_FRAME_MS / 1000.0))
    if frame_len < 160:  # minimal sinnvoll
        frame_len = 160
    max_frames = int(VAD_MAX_SECONDS * sr / frame_len) + 2
    min_frames = int(VAD_MIN_SECONDS * sr / frame_len)
    silence_needed_frames = int(VAD_END_HOLD * sr / frame_len)
    preroll_frames = int(VAD_PREROLL_SECONDS * sr / frame_len)

    collected = []
    peak_trace = []
    started = False
    silence_run = 0
    total_frames = 0
    start_time = time.time()

    stream = sd.InputStream(device=mic_id, samplerate=sr, channels=ch, dtype='int16', blocksize=frame_len)
    with stream:
        preroll = []
        while True:
            block, _ = stream.read(frame_len)
            total_frames += 1
            mono = _downmix_mono(block)
            mono16 = _resample(mono, sr, TARGET_STT_SR)
            peak_db = _analyze_peak_dbfs(mono16)
            peak_trace.append(peak_db)
            if not started:
                preroll.append(block.copy())
                if len(preroll) > preroll_frames:
                    preroll.pop(0)
                if peak_db >= VAD_START_DBFS:
                    started = True
                    collected.extend(preroll)
                    collected.append(block.copy())
                else:
                    if (time.time() - start_time) >= VAD_MAX_SECONDS:
                        return np.zeros((0,), dtype=np.int16), peak_trace
            else:
                collected.append(block.copy())
                if peak_db < VAD_END_DBFS:
                    silence_run += 1
                else:
                    silence_run = 0
                dur_so_far = len(collected) * frame_len / sr
                if dur_so_far >= VAD_MAX_SECONDS:
                    break
                if dur_so_far >= VAD_MIN_SECONDS and silence_run >= silence_needed_frames:
                    break
            if total_frames >= max_frames:
                break
    if not collected:
        return np.zeros((0,), dtype=np.int16), peak_trace
    audio = np.concatenate(collected, axis=0)
    return audio, peak_trace


def build_prompt(history: list[tuple[str,str]], user_text: str, keep: int) -> str:
    # history: list of (role, text) where role in {user, assistant}
    if keep>0:
        relevant = history[-keep:]
    else:
        relevant = history
    convo_lines = []
    for role, txt in relevant:
        tag = 'Benutzer' if role=='user' else 'Assistent'
        convo_lines.append(f"{tag}: {txt}")
    convo_str = "\n".join(convo_lines) if convo_lines else "(Kein bisheriger Kontext)"
    system_instruction = (
        "Du bist ein höflicher, knapper deutschsprachiger Assistent. Antworte prägnant (<=20 Wörter). "
        "Keine Wiederholung des Benutzertexts, keine Meta-Erklärungen."
    )
    return f"{system_instruction}\n\nBisheriges Gespräch:\n{convo_str}\n\nNeuer Benutzer: {user_text}\nAntwort:"  # Gemini erhält einen Gesamtprompt


def synth_and_play(text: str):
    tts_audio = synthesize_speech(text, sample_rate=TARGET_STT_SR)
    # Resample + Kanäle
    play = _resample(tts_audio, TARGET_STT_SR, audio_io.SAMPLE_RATE)
    if audio_io.CHANNELS>1:
        play = np.repeat(play[:,None], audio_io.CHANNELS, axis=1)
    audio_io.play_audio_chunk(play)


def run_conversation(args):
    if not GEMINI_API_KEY:
        print("[FEHLER] GEMINI_API_KEY nicht gesetzt.")
        return
    mic_id = _pick_device(MIC_DEVICE_NAME, 'input')
    spk_id = _pick_device(SPEAKER_DEVICE_NAME, 'output')
    print(f"Mikrofon: {mic_id}  Lautsprecher: {spk_id}")
    print("Starte Konversation. Sprich pro Runde klar. Stop-Wörter: " + ", ".join(sorted(STOP_WORDS)))

    history: list[tuple[str,str]] = []
    empty_streak = 0
    for turn in range(1, args.max_turns+1):
        if VAD_ENABLED:
            print(f"\n--- Runde {turn} (VAD aktiv) ---")
            try:
                raw, peak_trace = _record_vad_turn(mic_id)
            except Exception as e:
                print("[ERR] VAD-Aufnahmefehler:", e)
                break
            print(f"Aufnahme Dauer: {raw.shape[0]/audio_io.SAMPLE_RATE:.2f}s | Frames: {raw.shape[0]} | Peaks gespeichert: {len(peak_trace)}")
        else:
            print(f"\n--- Runde {turn} Aufnahme ({args.seconds:.1f}s) ---")
            frames = int(audio_io.SAMPLE_RATE * args.seconds)
            try:
                raw = sd.rec(frames, samplerate=audio_io.SAMPLE_RATE, channels=audio_io.CHANNELS, dtype='int16', device=mic_id)
                sd.wait()
            except Exception as e:
                print("[ERR] Aufnahmefehler:", e)
                break
        mono = _downmix_mono(raw)
        mono16 = _resample(mono, audio_io.SAMPLE_RATE, TARGET_STT_SR)
        peak_db = _analyze_peak_dbfs(mono16)
        print(f"Peak: {peak_db:.1f} dBFS")
        if peak_db < args.silence_dbfs:
            print(f"Stille (< {args.silence_dbfs} dBFS) -> übersprungen")
            empty_streak += 1
            if empty_streak >= args.max_empty:
                print("Zu viele leere Versuche -> Ende.")
                break
            continue
        # STT
        try:
            text = transcribe_audio(mono16, sample_rate=TARGET_STT_SR)
        except Exception as e:
            print("[ERR] STT:", e); empty_streak +=1; continue
        if not text:
            print("(leer)")
            empty_streak +=1
            if empty_streak >= args.max_empty:
                print("Zu viele leere Versuche -> Ende.")
                break
            continue
        empty_streak = 0
        print("User:", text)
        if text.lower().strip() in STOP_WORDS:
            print("Stop-Wort erkannt -> Konversation beendet.")
            break
        # Build prompt + Gemini
        prompt = build_prompt(history, text, args.context_turns)
        try:
            answer = get_gemini_response(prompt)
        except Exception as e:
            print("[ERR] Gemini:", e)
            break
        print("Assistent:", answer)
        # Update history
        history.append(('user', text))
        history.append(('assistant', answer))
        # TTS
        try:
            synth_and_play(answer)
        except Exception as e:
            print("[ERR] TTS/Wiedergabe:", e)
            break
    print("\nKonversation beendet.")


def main():
    p = argparse.ArgumentParser(description="Multi-Turn Sprach-Konversationstest")
    p.add_argument('--seconds', type=float, default=4.0)
    p.add_argument('--max-turns', type=int, default=10)
    p.add_argument('--context-turns', type=int, default=6)
    p.add_argument('--silence-dbfs', type=float, default=-45.0)
    p.add_argument('--max-empty', type=int, default=3)
    # Hinweis: VAD Parameter jetzt in config.py (VAD_ENABLED, VAD_START_DBFS, ...)
    args = p.parse_args()
    run_conversation(args)

if __name__ == '__main__':  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        print('\nAbbruch durch Benutzer')
        sys.exit(1)
