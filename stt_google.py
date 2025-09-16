"""Google Cloud Speech-to-Text Modul

Vereinfachte Variante: Es wird ausschließlich die Umgebungsvariable
`GOOGLE_APPLICATION_CREDENTIALS` erwartet, die auf die Service-Account
JSON-Datei zeigt. Keine alternative Inline-JSON Logik mehr.
"""

from google.cloud import speech
from config import GOOGLE_APPLICATION_CREDENTIALS
import os

# Stellt sicher, dass die Variable für die Google SDKs verfügbar ist
if GOOGLE_APPLICATION_CREDENTIALS:
    if os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
    else:
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS zeigt auf keine existierende Datei: {GOOGLE_APPLICATION_CREDENTIALS}"
        )
else:
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS ist nicht gesetzt")

"""Google Speech-to-Text: Wandelt Audio (int16 numpy array) in Text um."""

def transcribe_audio(audio_data, sample_rate=16000):
    """
    Sendet Audiodaten an die Google Speech-to-Text API und gibt das Transkript zurück.
    audio_data: numpy array (int16), mono
    """
    client = speech.SpeechClient()
    audio_bytes = audio_data.tobytes()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="de-DE",
        audio_channel_count=1,
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript
    return transcript.strip()
