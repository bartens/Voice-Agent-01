"""
Gemini API Modul
"""

import google.generativeai as genai
from config import GEMINI_MODEL_NAME, GEMINI_API_KEY

def _pick_first_text(resp) -> str:
    try:
        return (resp.text or "").strip()
    except Exception:
        # Fallback manuelle Extraktion
        try:
            parts = []
            for cand in resp.candidates or []:
                for part in cand.content.parts:
                    if hasattr(part, 'text'):
                        parts.append(part.text)
            return "\n".join(p.strip() for p in parts if p.strip())
        except Exception:
            return ""

def get_gemini_response(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY nicht gesetzt (System-Umgebungsvariable fehlt)")
    genai.configure(api_key=GEMINI_API_KEY)
    model_name = GEMINI_MODEL_NAME
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        txt = _pick_first_text(resp)
        if not txt:
            return "(Leere Antwort)"
        return txt
    except Exception as e:
        msg = str(e)
        if "404" in msg and "not found" in msg.lower():
            # Hilfe anbieten
            try:
                models = genai.list_models()
                names = [m.name for m in models if hasattr(m, 'name')]
            except Exception:
                names = []
            raise RuntimeError(
                f"Gemini Modell '{model_name}' nicht verfügbar. Verfügbar (Auszug): {names[:8]} - setze env GEMINI_MODEL_NAME oder passe config an. Originalfehler: {msg}"
            ) from e
        raise
