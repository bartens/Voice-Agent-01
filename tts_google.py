"""
Google Cloud Text-to-Speech Modul
"""

from google.cloud import texttospeech
from config import (
    GOOGLE_APPLICATION_CREDENTIALS,
    TTS_LANGUAGE_CODE,
    TTS_VOICE_NAME,
    TTS_SSML_GENDER,
    TTS_SPEAKING_RATE,
    TTS_PITCH,
    TTS_VOLUME_GAIN_DB,
)
import os

# Setze die Umgebungsvariable für die Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

"""
Google Text-to-Speech Modul: Wandelt Text in Audio um.
"""

import numpy as np

GENDER_MAP = {
    "NEUTRAL": texttospeech.SsmlVoiceGender.NEUTRAL,
    "FEMALE": texttospeech.SsmlVoiceGender.FEMALE,
    "MALE": texttospeech.SsmlVoiceGender.MALE,
}


def list_voices(language_code: str | None = None):
    """Hilfsfunktion: verfügbare Stimmen auflisten (optional)."""
    client = texttospeech.TextToSpeechClient()
    resp = client.list_voices(language_code=language_code)
    out = []
    for v in resp.voices:
        out.append({
            "name": v.name,
            "lang_codes": list(v.language_codes),
            "gender": texttospeech.SsmlVoiceGender(v.ssml_gender).name,
            "natural_sample_rate_hz": v.natural_sample_rate_hertz,
        })
    return out


def synthesize_speech(text, sample_rate=16000):
    """
    Sendet Text an die Google TTS API und gibt Audiodaten (int16 numpy array) zurück.
    """
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    if TTS_VOICE_NAME:
        voice = texttospeech.VoiceSelectionParams(
            name=TTS_VOICE_NAME,
            language_code=TTS_LANGUAGE_CODE,
        )
    else:
        voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE_CODE,
            ssml_gender=GENDER_MAP.get(TTS_SSML_GENDER, texttospeech.SsmlVoiceGender.NEUTRAL),
        )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        speaking_rate=TTS_SPEAKING_RATE,
        pitch=TTS_PITCH,
        volume_gain_db=TTS_VOLUME_GAIN_DB,
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    audio_array = np.frombuffer(response.audio_content, dtype=np.int16)
    return audio_array
