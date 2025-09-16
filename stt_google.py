"""
Google Cloud Speech-to-Text Modul
"""

from google.cloud import speech
import numpy as np
from config import GOOGLE_APPLICATION_CREDENTIALS
import os

# Setze die Umgebungsvariable für die Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

"""
Google Speech-to-Text Modul: Wandelt Audio in Text um.
"""

import os
from google.cloud import speech
import numpy as np

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
