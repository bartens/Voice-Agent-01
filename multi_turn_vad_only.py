"""Kontinuierliche Konversation rein über VAD (kein fester Rundenrahmen)

Verhalten:
 - Endlosschleife: wartet auf Sprachbeginn (VAD_START_DBFS), zeichnet auf bis Sprachende (VAD_END_DBFS & VAD_END_HOLD)
 - Jeder erkannte Sprachabschnitt => STT -> Kontext-Prompt -> Gemini -> TTS Antwort
 - Stop-Wort (stop/ende/quit/tschüss) beendet Schleife
 - Keine feste Turn-Zahl; Abbruch per CTRL+C

Nutzen: Flüssiger als Round-Based Ansatz, weil Stille zwischen Turns nicht künstlich verschwendet wird.

Konfiguration ausschließlich über config.py (VAD_* und Geräte, Modell, etc.).
"""
from __future__ import annotations
import time
import numpy as np
import sounddevice as sd
import sys

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
import audio_io
from stt_google import transcribe_audio
from gemini_api import get_gemini_response
from tts_google import synthesize_speech

TARGET_STT_SR = 16000
STOP_WORDS = {"stop", "ende", "quit", "tschüss", "tschuss"}

def _pick_device(name_or_id, kind):
    devices = sd.query_devices()
    if isinstance(name_or_id, int):
        return name_or_id
    low = str(name_or_id).lower()
    # exact
    for i,d in enumerate(devices):
        if d['name'].lower()==low and d[f'max_{kind}_channels']>0:
            return i
    # substring
    for i,d in enumerate(devices):
        if low in d['name'].lower() and d[f'max_{kind}_channels']>0:
            return i
    di,do = sd.default.device
    return di if kind=='input' else do

def _downmix_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim==1: return arr.astype(np.int16)
    m = arr.mean(axis=1)
    return np.clip(m, -32768, 32767).astype(np.int16)

def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr==target_sr or sig.size==0: return sig
    x = sig.astype(np.float32)
    n_new = int(round(x.shape[0]*target_sr/orig_sr))
    if n_new<2: return sig[:1]
    idx = np.linspace(0, x.shape[0]-1, n_new)
    out = np.interp(idx, np.arange(x.shape[0]), x)
    return np.clip(out, -32768, 32767).astype(np.int16)

def _peak_dbfs(samples: np.ndarray) -> float:
    if samples.size==0: return -120.0
    p = np.max(np.abs(samples))
    if p==0: return -120.0
    return 20*np.log10(p/32767.0)

def _tts_play(text: str):
    audio = synthesize_speech(text, sample_rate=TARGET_STT_SR)
    play = _resample(audio, TARGET_STT_SR, audio_io.SAMPLE_RATE)
    if audio_io.CHANNELS>1:
        play = np.repeat(play[:,None], audio_io.CHANNELS, axis=1)
    audio_io.play_audio_chunk(play)

def _record_segment(mic_id: int):
    sr = audio_io.SAMPLE_RATE
    ch = audio_io.CHANNELS
    frame_len = int(sr * (VAD_FRAME_MS/1000.0))
    if frame_len<160: frame_len=160
    max_frames = int(VAD_MAX_SECONDS * sr / frame_len) + 2
    min_frames = int(VAD_MIN_SECONDS * sr / frame_len)
    silent_needed = int(VAD_END_HOLD * sr / frame_len)
    preroll_frames = int(VAD_PREROLL_SECONDS * sr / frame_len)

    stream = sd.InputStream(device=mic_id, samplerate=sr, channels=ch, dtype='int16', blocksize=frame_len)
    collected = []
    preroll = []
    started=False
    silence_run=0
    with stream:
        for idx in range(max_frames):
            block, _ = stream.read(frame_len)
            mono = _downmix_mono(block)
            mono16 = _resample(mono, sr, TARGET_STT_SR)
            peak = _peak_dbfs(mono16)
            if not started:
                preroll.append(block.copy())
                if len(preroll)>preroll_frames:
                    preroll.pop(0)
                if peak >= VAD_START_DBFS:
                    started=True
                    collected.extend(preroll)
                    collected.append(block.copy())
            else:
                collected.append(block.copy())
                if peak < VAD_END_DBFS:
                    silence_run +=1
                else:
                    silence_run = 0
                dur = len(collected)*frame_len/sr
                if dur >= VAD_MAX_SECONDS:
                    break
                if dur >= VAD_MIN_SECONDS and silence_run >= silent_needed:
                    break
    if not collected:
        return np.zeros((0,), dtype=np.int16)
    return np.concatenate(collected, axis=0)

def run_loop():
    if not GEMINI_API_KEY:
        print("[FEHLER] GEMINI_API_KEY fehlt")
        return
    if not VAD_ENABLED:
        print("[HINWEIS] VAD_ENABLED=False in config.py -> aktiviere für dieses Skript")
        return
    mic_id = _pick_device(MIC_DEVICE_NAME, 'input')
    spk_id = _pick_device(SPEAKER_DEVICE_NAME, 'output')
    print(f"VAD Konversation gestartet | Mic {mic_id} | Spk {spk_id}")
    history: list[tuple[str,str]] = []
    turn=0
    try:
        while True:
            print("\n[WARTEN] auf Sprache ... (CTRL+C für Ende)")
            segment = _record_segment(mic_id)
            if segment.size==0:
                continue
            turn+=1
            mono = _downmix_mono(segment)
            mono16 = _resample(mono, audio_io.SAMPLE_RATE, TARGET_STT_SR)
            peak = _peak_dbfs(mono16)
            print(f"Turn {turn} Dauer {mono.size/audio_io.SAMPLE_RATE:.2f}s Peak {peak:.1f} dBFS")
            # STT
            try:
                text = transcribe_audio(mono16, sample_rate=TARGET_STT_SR)
            except Exception as e:
                print("[ERR] STT:", e); continue
            if not text:
                print("(leer)"); continue
            print("User:", text)
            if text.lower().strip() in STOP_WORDS:
                print("Stop-Wort erkannt -> Ende")
                break
            # Prompt bauen
            short_hist = history[-8:]
            convo = []
            for role,msg in short_hist:
                tag = 'Benutzer' if role=='user' else 'Assistent'
                convo.append(f"{tag}: {msg}")
            convo_str = "\n".join(convo) if convo else "(kein Kontext)"
            prompt = (
                "Du bist ein sehr knapper deutschsprachiger Assistent (<=18 Wörter). "
                "Vermeide Wiederholungen.\n\nKontext:\n" + convo_str + f"\n\nNeuer Benutzer: {text}\nAntwort:" )
            try:
                answer = get_gemini_response(prompt)
            except Exception as e:
                print("[ERR] Gemini:", e); continue
            print("Assistent:", answer)
            history.append(('user', text))
            history.append(('assistant', answer))
            try:
                _tts_play(answer)
            except Exception as e:
                print("[ERR] TTS:", e)
    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer")
    print("Konversation beendet.")

def main():
    run_loop()

if __name__ == '__main__':  # pragma: no cover
    main()