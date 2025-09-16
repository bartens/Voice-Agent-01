"""
Konfigurationsdatei für API-Keys und Audio-Settings
"""

import os

# Gemini API Key (nur aus System-Umgebungsvariable: GEMINI_API_KEY)
# Kein Fallback / kein Hardcoding hier, damit kein versehentliches Commit von Secrets passiert.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Pfad zur Google Credentials Datei aus Umgebungsvariable
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Virtuelle Audiokabel (aus Screenshot)
SPEAKER_DEVICE_NAME = 8 # 6
MIC_DEVICE_NAME = 4 # 1

# Gemini Modell (optional per Umgebungsvariable überschreibbar)
# Empfohlene Werte (Stand 2025): "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest"
# Gemini Modell direkt hier eintragen (kein Env-Fallback nötig)
# Beispiele: "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Aufnahmedauer pro STT-Chunk (Sekunden) – kann bei Bedarf erhöht werden (z.B. 3.0 oder 5.0)
STT_CHUNK_SECONDS = float(os.getenv("STT_CHUNK_SECONDS", "5.0"))

# Google Calendar (Function Calling Tools) – Standard-Kalender-ID
CALENDAR_ID = os.getenv("CALENDAR_ID", "primary")

# OAuth Client Secret / Token Pfade für Google Calendar (zentralisiert)
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "client_secret.json")
GOOGLE_OAUTH_TOKEN = os.getenv("GOOGLE_OAUTH_TOKEN", "token.json")


# --- VAD Einstellungen (zentral für alle Module) ---
# Aktivierung
VAD_ENABLED = os.getenv("VAD_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
# Startschwelle (Peak dBFS) ab der Sprache erkannt wird
VAD_START_DBFS = float(os.getenv("VAD_START_DBFS", "-45.0"))
# Stille-Schwelle (Peak dBFS) unterhalb der als potenzielles Ende gezählt wird
VAD_END_DBFS = float(os.getenv("VAD_END_DBFS", "-50.0"))
# Benötigte Stille-Dauer nach Sprachende (Sekunden)
VAD_END_HOLD = float(os.getenv("VAD_END_HOLD", "0.6"))
# Minimale und maximale Gesamtdauer der Aufnahme
VAD_MIN_SECONDS = float(os.getenv("VAD_MIN_SECONDS", "0.6"))
VAD_MAX_SECONDS = float(os.getenv("VAD_MAX_SECONDS", "8.0"))
# Frame-Größe für Analyse (ms)
VAD_FRAME_MS = float(os.getenv("VAD_FRAME_MS", "30.0"))
# Pre-Roll (Sekunden, werden vor erkannter Start-Schwelle gepuffert und angehängt)
VAD_PREROLL_SECONDS = float(os.getenv("VAD_PREROLL_SECONDS", "0.2"))

# --- TTS Einstellungen ---
# Sprache (z.B. "de-DE", "en-US")
TTS_LANGUAGE_CODE = os.getenv("TTS_LANGUAGE_CODE", "de-DE")
# Konkrete Voice (optional). Beispiel: "de-DE-Wavenet-F" oder "de-DE-Neural2-F".
# Vorgabe: "de-DE-Chirp3-HD-Kore"; kann per ENV TTS_VOICE_NAME überschrieben werden.
TTS_VOICE_NAME = os.getenv("TTS_VOICE_NAME", "de-DE-Chirp3-HD-Aoede")
# SSML Gender (NEUTRAL, FEMALE, MALE) – wird ignoriert, wenn TTS_VOICE_NAME gesetzt ist
TTS_SSML_GENDER = os.getenv("TTS_SSML_GENDER", "NEUTRAL").upper()
# Sprechgeschwindigkeit (1.0 = normal)
TTS_SPEAKING_RATE = float(os.getenv("TTS_SPEAKING_RATE", "1.0"))
# Tonhöhe in Halbton-Schritten (z.B. -2.0 bis +2.0)
TTS_PITCH = float(os.getenv("TTS_PITCH", "0.0"))
# Lautstärke-Gewinn in dB (-96.0 bis +16.0)
TTS_VOLUME_GAIN_DB = float(os.getenv("TTS_VOLUME_GAIN_DB", "0.0"))

# --- SIP Einstellungen (für Variante 3: direkter SIP-Stack) ---
# SIP Registrar / Server (z.B. sip.example.com oder IP)
SIP_REGISTRAR = os.getenv("SIP_REGISTRAR", "")
# Benutzername / Extension
SIP_USERNAME = os.getenv("SIP_USERNAME", "")
# Passwort
SIP_PASSWORD = os.getenv("SIP_PASSWORD", "")
# SIP Domain (falls vom Registrar verschieden nötig)
SIP_DOMAIN = os.getenv("SIP_DOMAIN", SIP_REGISTRAR)
# Lokaler UDP Port (0 = automatisch wählen)
SIP_LOCAL_PORT = int(os.getenv("SIP_LOCAL_PORT", "0"))
# Auto-Answer aktivieren
SIP_AUTO_ANSWER = os.getenv("SIP_AUTO_ANSWER", "true").lower() in {"1","true","yes","on"}
# Automatische Begrüßung per TTS nach Verbindungsaufbau
SIP_GREETING_ENABLED = os.getenv("SIP_GREETING_ENABLED", "true").lower() in {"1","true","yes","on"}
# Begrüßungstext (Fallback wenn Persönlichkeit/Prompt nicht geladen wird)
SIP_GREETING_TEXT = os.getenv("SIP_GREETING_TEXT", "Guten Tag. Einen Moment bitte.")
# Maximale einfache Aufnahme-Dauer für erste Nutzeräußerung (Sekunden)
SIP_INITIAL_CAPTURE_SECONDS = float(os.getenv("SIP_INITIAL_CAPTURE_SECONDS", "6.0"))

