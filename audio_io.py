"""
Audio I/O Modul: Liest und spielt Audio über das virtuelle Audiokabel.
"""



import sounddevice as sd
import numpy as np
from config import SPEAKER_DEVICE_NAME, MIC_DEVICE_NAME, STT_CHUNK_SECONDS

# Hinweis: Gerätekatalog-Ausgabe entfernt (Anforderung). Falls Debug nötig:
# Einfach temporär folgende Zeilen aktivieren:
#   for i, dev in enumerate(sd.query_devices()):
#       print(i, dev['name'])

SAMPLE_RATE = 48000
CHANNELS = 2
CHUNK_DURATION = STT_CHUNK_SECONDS  # Sekunden (aus config)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

def _get_device_id(device_name, kind='input'):
    devices = sd.query_devices()
    # Wenn device_name eine int ist, direkt als Geräte-ID verwenden
    if isinstance(device_name, int):
        return device_name
    # Wenn device_name "Speakers" oder "Microphone" ist, verwende das Standardgerät
    if isinstance(device_name, str):
        if device_name.lower() == "speakers":
            return sd.default.device[1]  # Output
        if device_name.lower() == "microphone":
            return sd.default.device[0]  # Input
        for idx, dev in enumerate(devices):
            if device_name.lower() in dev['name'].lower() and dev['max_' + kind + '_channels'] > 0:
                return idx
    raise RuntimeError(f"Audiogerät '{device_name}' nicht gefunden.")

def read_audio_chunk():
    """Liest einen Audio-Chunk vom virtuellen Kabel (Mikrofon)."""
    device_id = _get_device_id(MIC_DEVICE_NAME, kind='input')
    audio = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', device=device_id)
    sd.wait()
    return audio.flatten()

def play_audio_chunk(audio_data):
    """Spielt einen Audio-Chunk auf dem virtuellen Kabel (Lautsprecher) ab."""
    device_id = _get_device_id(SPEAKER_DEVICE_NAME, kind='output')
    # Falls audio_data als bytes kommt, in numpy umwandeln
    if isinstance(audio_data, bytes):
        audio_data = np.frombuffer(audio_data, dtype='int16')
    # Form in (frames, channels) bringen, da sounddevice 'channels' Parameter hier nicht benötigt
    if audio_data.ndim == 1:
        if CHANNELS > 1:
            # Prüfen ob Länge durch CHANNELS teilbar ist (interleaved?)
            if audio_data.size % CHANNELS == 0:
                reshaped = audio_data.reshape(-1, CHANNELS)
            else:
                # Replizieren zu Mehrkanal
                reshaped = np.repeat(audio_data[:, None], CHANNELS, axis=1)
            audio_data = reshaped
    sd.play(audio_data, samplerate=SAMPLE_RATE, device=device_id)
    sd.wait()
