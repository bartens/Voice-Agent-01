"""Virtual Cable Listener

Startet den Kalender-Sprachassistenten on-demand, wenn eingehendes Audio (Anruf über MicroSIP Auto-Answer) eine Aktivität überschreitet
oder ein optionales Hotword erkannt wird. Primär für Nutzung mit virtuellem Audiokabel.

Modi:
 1) Energy Trigger: Sobald länger als TRIGGER_MIN_SECONDS Sprachpegel > TRIGGER_DBFS anliegt und der Assistent gerade nicht läuft.
 2) (Optional) Hotword: Einfache Stichwort-Erkennung aus STT-Teilsegmenten (z.B. "assistent"), bevor voller Loop startet.

Nach Trigger wird `speech_calendar_assistant.run()` gestartet (blockierend) und nach dessen Ende kehrt der Listener zurück.

Abbruch: CTRL+C
"""
from __future__ import annotations
import time
import threading
import numpy as np
import sounddevice as sd

import audio_io
from speech_calendar_assistant import run as run_calendar_assistant
from config import (
    MIC_DEVICE_NAME,
    VAD_START_DBFS,
    VAD_FRAME_MS,
)

# Konfiguration Listener
TRIGGER_DBFS = max(VAD_START_DBFS - 5, -50)  # etwas sensibler als Haupt-VAD
TRIGGER_MIN_SECONDS = 0.4                    # Mindestdauer über Schwellwert
COOLDOWN_SECONDS = 3.0                       # Nach Ende Assistent warte bevor neu getriggert wird
HOTWORD_ENABLED = False
HOTWORD = "assistent"  # Placeholder (wenn aktiviert muss Mini-STT implementiert werden)
PRINT_LEVEL = True

_running_assistant = threading.Event()
_stop_flag = threading.Event()


def _pick_input_device(name_or_id):
    devices = sd.query_devices()
    if isinstance(name_or_id, int):
        return name_or_id
    low = str(name_or_id).lower()
    for i, d in enumerate(devices):
        if low in d['name'].lower() and d['max_input_channels'] > 0:
            return i
    return sd.default.device[0]


def _peak_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    peak = np.max(np.abs(samples))
    if peak <= 0:
        return -120.0
    return 20 * np.log10(peak / 32767.0)


def _assistant_wrapper():
    try:
        run_calendar_assistant()
    finally:
        _running_assistant.clear()
        time.sleep(COOLDOWN_SECONDS)


def start_listener():
    dev_id = _pick_input_device(MIC_DEVICE_NAME)
    sr = audio_io.SAMPLE_RATE
    ch = audio_io.CHANNELS
    frame_len = int(sr * (VAD_FRAME_MS / 1000.0))
    hold_frames = int(TRIGGER_MIN_SECONDS * sr / frame_len)
    if hold_frames <= 0:
        hold_frames = 1
    print(f"[LISTENER] Device={dev_id} Trigger={TRIGGER_DBFS}dBFS hold={TRIGGER_MIN_SECONDS}s")
    print("[LISTENER] Warte auf eingehende Sprache... (CTRL+C zum Abbruch)")

    stream = sd.InputStream(device=dev_id, channels=ch, samplerate=sr, dtype='int16', blocksize=frame_len)
    above = 0
    last_level_print = 0
    with stream:
        while not _stop_flag.is_set():
            block, _ = stream.read(frame_len)
            if block is None:
                continue
            mono = block if block.ndim == 1 else block.mean(axis=1).astype(np.int16)
            peak = _peak_dbfs(mono)
            if PRINT_LEVEL and (time.time() - last_level_print) > 1.0:
                print(f"[LEVEL] {peak:6.1f} dBFS")
                last_level_print = time.time()
            if peak >= TRIGGER_DBFS:
                above += 1
            else:
                above = 0
            if (not _running_assistant.is_set()) and above >= hold_frames:
                print("[LISTENER] Trigger erkannt -> Starte Assistent")
                _running_assistant.set()
                threading.Thread(target=_assistant_wrapper, daemon=True).start()
                above = 0
            time.sleep(0)  # yield


def main():
    try:
        start_listener()
    except KeyboardInterrupt:
        print("\n[LISTENER] Abbruch")
        _stop_flag.set()


if __name__ == "__main__":  # pragma: no cover
    main()
