"""Hilfsfunktionen zur Bereinigung von Markdown-/Listenformatierung aus Modell-Antworten.

sanitize_output(text):
  - Entfernt **Fett** -> Fett
  - Entfernt führende Aufzählungssternchen/-striche/•
  - Reduziert Mehrfachleerzeilen
  - Trimmt
"""
from __future__ import annotations
import re
from typing import Optional

def sanitize_output(text: Optional[str]) -> str:
    if not text:
        return ''
    # Entferne Markdown Bold **...**
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Entferne Inline Code-Markierungen `...`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Zeilenweise Bullet-Symbole am Anfang entfernen
    cleaned_lines = []
    for line in text.splitlines():
        cleaned = re.sub(r'^\s*([\-*•]+)\s+', '', line)
        # Falls der gesamte Rest nur aus Sternchen bestand -> leer
        if re.fullmatch(r'[\-*•]+', cleaned.strip()):
            cleaned = ''
        cleaned_lines.append(cleaned)
    cleaned = '\n'.join(cleaned_lines)
    # Mehrfach Leerzeilen reduzieren
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

__all__ = ["sanitize_output"]
