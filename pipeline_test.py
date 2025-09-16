"""Pipeline Test Script

Ablauf (orientiert am Vorgehen des vereinfachten Audio-Tests):
 1. Geräte aus config auswählen
 2. Audio für definierte Dauer aufnehmen
 3. Analyse (Pegel / Samples)
 4. Downmix + Resample -> 16 kHz mono für STT
 5. Google Speech-to-Text
 6. Gemini Antwort erzeugen (knapp, deutsch)
 7. Google Text-to-Speech (16 kHz mono)
 8. Resample auf Wiedergabe-Samplerate + Kanal-Anpassung
 9. Wiedergabe

Voraussetzungen:
 - Umgebungsvariablen GOOGLE_APPLICATION_CREDENTIALS, GEMINI_API_KEY gesetzt
 - Netzwerkzugriff für Google APIs
 - sounddevice funktionierende Geräte (config)

Aufruf:
  python pipeline_test.py --seconds 4
Optional:
  --raw-transcript  (zeigt Roh-Transkript auch bei leerem Ergebnis)

Hinweis: Dieses Script führt einen einzelnen End-to-End Durchlauf aus.
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import sounddevice as sd

from config import MIC_DEVICE_NAME, SPEAKER_DEVICE_NAME
import audio_io  # für SAMPLE_RATE, CHANNELS und play_audio_chunk
from stt_google import transcribe_audio
from gemini_api import get_gemini_response
from tts_google import synthesize_speech

TARGET_STT_SR = 16000  # Google STT / TTS Standard in unserem Setup


def _pick_device(requested: str | int, kind: str) -> int:
    devices = sd.query_devices()
    if isinstance(requested, int):
        if 0 <= requested < len(devices):
            if devices[requested][f'max_{kind}_channels'] > 0:
                return requested
            raise RuntimeError(f"Gerät-ID {requested} hat keine {kind}-Kanäle")
        raise RuntimeError(f"Gerät-ID {requested} existiert nicht")
    req_lower = str(requested).lower()
    exact = [i for i,d in enumerate(devices) if d['name'].lower() == req_lower and d[f'max_{kind}_channels']>0]
    if exact:
        return exact[0]
    subs = [i for i,d in enumerate(devices) if req_lower in d['name'].lower() and d[f'max_{kind}_channels']>0]
    if subs:
        return subs[0]
    def_in, def_out = sd.default.device
    fallback = def_in if kind=='input' else def_out
    print(f"[WARN] Kein Treffer für '{requested}' ({kind}); Fallback -> {fallback}:{devices[fallback]['name']}")
    return fallback


def _analyze(data: np.ndarray, label: str):
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data
    peak = int(np.max(np.abs(mono))) if mono.size else 0
    rms = float(np.sqrt(np.mean(mono.astype(np.float32)**2))) if mono.size else 0.0
    dbfs = -120.0 if peak == 0 else 20 * np.log10(peak/32767.0)
    print(f"Analyse {label}: samples={mono.size} peak={peak} rms={rms:.1f} dBFS={dbfs:.1f}")


def _downmix_mono_int16(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.int16)
    mono = data.mean(axis=1)
    mono = np.clip(mono, -32768, 32767)
    return mono.astype(np.int16)


def _resample_linear_int16(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    if data.size == 0:
        return data
    # convert to float for interpolation
    x = data.astype(np.float32)
    n_orig = x.shape[0]
    n_new = int(round(n_orig * target_sr / orig_sr))
    if n_new <= 1:
        return data[:1]
    src_idx = np.linspace(0, n_orig - 1, num=n_new)
    base_idx = np.arange(n_orig)
    resampled = np.interp(src_idx, base_idx, x)
    resampled = np.clip(resampled, -32768, 32767)
    return resampled.astype(np.int16)


def run_pipeline(seconds: float, raw_transcript: bool):
    # Vorbedingungen prüfen
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("[WARN] GOOGLE_APPLICATION_CREDENTIALS nicht gesetzt")
    if not os.getenv("GEMINI_API_KEY"):
        print("[WARN] GEMINI_API_KEY nicht gesetzt")

    print("Config Geräte:", MIC_DEVICE_NAME, "/", SPEAKER_DEVICE_NAME)
    mic_id = _pick_device(MIC_DEVICE_NAME, 'input')
    spk_id = _pick_device(SPEAKER_DEVICE_NAME, 'output')
    mic_info = sd.query_devices(mic_id)
    spk_info = sd.query_devices(spk_id)
    print(f"Mikrofon: {mic_id} -> {mic_info['name']}")
    print(f"Lautsprecher: {spk_id} -> {spk_info['name']}")

    # Aufnahme (48000 Hz stereo wie audio_io)
    sr_cap = audio_io.SAMPLE_RATE
    ch_cap = audio_io.CHANNELS
    frames = int(sr_cap * seconds)
    print(f"Aufnahme startet ({seconds:.1f}s @ {sr_cap} Hz, ch={ch_cap}) – jetzt sprechen...")
    try:
        rec = sd.rec(frames, samplerate=sr_cap, channels=ch_cap, dtype='int16', device=mic_id)
        sd.wait()
    except Exception as e:
        print("[ERR] Aufnahme fehlgeschlagen:", e)
        return
    _analyze(rec, "Capture")

    # Downmix + Resample für STT
    mono16 = _downmix_mono_int16(rec)
    stt_input = _resample_linear_int16(mono16, sr_cap, TARGET_STT_SR)
    _analyze(stt_input, "STT-Input (16k mono)")

    # STT
    print("Sende an Speech-to-Text...")
    try:
        transcript = transcribe_audio(stt_input, sample_rate=TARGET_STT_SR)
    except Exception as e:
        print("[ERR] STT fehlgeschlagen:", e)
        return
    if transcript:
        print("Transkript:", transcript)
    else:
        print("[WARN] Leeres Transkript")
        if raw_transcript:
            print("(Roh leer)")

    # Falls nichts erkannt wurde, abbrechen (oder Default Prompt)
    if not transcript:
        transcript = "(Kein Sprachinhalt erkannt)"

    # Gemini
    prompt = f"Antworte sehr kurz (max 12 Wörter) und höflich auf Deutsch auf folgendes: {transcript}"
    print("Sende an Gemini...")
    try:
        gemini_answer = get_gemini_response(prompt)
    except Exception as e:
        print("[ERR] Gemini fehlgeschlagen:", e)
        return
    print("Gemini Antwort:", gemini_answer)

    # TTS (16k mono)
    print("Synthese (TTS)...")
    try:
        tts_audio = synthesize_speech(gemini_answer, sample_rate=TARGET_STT_SR)
    except Exception as e:
        print("[ERR] TTS fehlgeschlagen:", e)
        return
    _analyze(tts_audio, "TTS 16k mono")

    # Resample + Kanal-Anpassung für Playback (audio_io erwartet SAMPLE_RATE & CHANNELS)
    play_audio = _resample_linear_int16(tts_audio, TARGET_STT_SR, audio_io.SAMPLE_RATE)
    if audio_io.CHANNELS > 1:
        play_audio = np.repeat(play_audio[:, None], audio_io.CHANNELS, axis=1)
    print("Wiedergabe Antwort...")
    try:
        audio_io.play_audio_chunk(play_audio)
    except Exception as e:
        print("[ERR] Wiedergabe fehlgeschlagen:", e)
        return
    print("Pipeline abgeschlossen.")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Test (Audio -> STT -> Gemini -> TTS -> Playback)")
    parser.add_argument('--seconds', type=float, default=4.0, help='Aufnahmedauer in Sekunden')
    parser.add_argument('--raw-transcript', action='store_true', help='Rohes (auch leeres) Transkript anzeigen')
    args = parser.parse_args()
    run_pipeline(args.seconds, args.raw_transcript)


if __name__ == '__main__':  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        print('\nAbbruch durch Benutzer')
        sys.exit(1)
