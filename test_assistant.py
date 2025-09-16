"""
Testskript für den Telefonassistenten
Nimmt Audio auf, transkribiert, sendet an Gemini, wandelt Antwort in Sprache und spielt sie ab.
"""

from audio_io import read_audio_chunk, play_audio_chunk
from stt_google import transcribe_audio
from gemini_api import get_gemini_response
from tts_google import synthesize_speech

print("Starte Test: Bitte sprich nach dem Start einen Satz ins Mikrofon...")

# Schritt 1: Audio aufnehmen
audio = read_audio_chunk()
print("Audio aufgenommen.")

# Schritt 2: Transkribieren
transcript = transcribe_audio(audio)
print(f"Transkript: {transcript}")

# Schritt 3: Gemini-Antwort holen
response = get_gemini_response(transcript)
print(f"Gemini-Antwort: {response}")

# Schritt 4: Antwort synthetisieren
audio_response = synthesize_speech(response)
print("Antwort wird abgespielt...")
play_audio_chunk(audio_response)
print("Test abgeschlossen.")
