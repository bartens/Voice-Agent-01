from audio_io import read_audio_chunk, play_audio_chunk
from stt_google import transcribe_audio
from gemini_api import get_gemini_response
from tts_google import synthesize_speech


def run_assistant():
    while True:
        audio = read_audio_chunk()
        transcript = transcribe_audio(audio)
        response = get_gemini_response(transcript)
        audio_response = synthesize_speech(response)
        play_audio_chunk(audio_response)


if __name__ == "__main__":
    run_assistant()
