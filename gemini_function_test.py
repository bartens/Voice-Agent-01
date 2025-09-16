"""Stiller Test-Harness für gemini_function_chat.

Zweck:
 - Import ohne sofortiges Initialisieren des Modells (dank Lazy Init im Hauptmodul)
 - Einfacher Aufruf von handle_user_message für manuelle oder automatisierte Smoke-Tests

Nutzung:
  python gemini_function_test.py "Liste meine Termine morgen"
Oder ohne Argument -> interaktiv eine einzige Anfrage.

Kein dauerhaftes Chat-Protokoll; jeder Aufruf separat.
"""
from __future__ import annotations
import sys
from gemini_function_chat import handle_user_message  # Lazy init erst bei Aufruf


def main():  # pragma: no cover
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        try:
            prompt = input("Prompt: ").strip()
        except EOFError:
            return
    if not prompt:
        print("(leer)")
        return
    try:
        answer = handle_user_message(prompt)
    except Exception as e:
        print("Fehler:", e)
        return
    print("Antwort:", answer)


if __name__ == "__main__":
    main()
