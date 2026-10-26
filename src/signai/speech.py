"""Text-to-speech and clipboard helpers.

* Browser TTS (Web Speech API) - used by the Streamlit app; no server-side
  dependency, works locally and on Streamlit Community Cloud.
* gTTS - server-side fallback that returns audio bytes for ``st.audio``.
* pyttsx3 - fully offline TTS for the local CLI demos (Windows SAPI / espeak).
"""
from __future__ import annotations

import io
import json


def browser_speak_html(text: str, lang: str = "en-US", rate: float = 1.05) -> str:
    """Self-contained HTML that speaks ``text`` in the browser.

    Pass the result to ``st.components.v1.html(..., height=0)``.
    """
    payload = json.dumps({"text": text, "lang": lang, "rate": rate})
    js = (
        "(function(){"
        "try{var p=" + payload + ";"
        "var synth=window.speechSynthesis;if(!synth){return;}"
        "synth.cancel();"
        "var u=new SpeechSynthesisUtterance(p.text);"
        "u.lang=p.lang;u.rate=p.rate;"
        "synth.speak(u);"
        "}catch(e){console.warn('TTS failed:',e);}})();"
    )
    return f"<script>{js}</script>"


def copy_to_clipboard_html(text: str) -> str:
    """Self-contained HTML that copies ``text`` to the clipboard."""
    payload = json.dumps(text)
    js = (
        "(function(){"
        "var t=" + payload + ";"
        "function fallback(){"
        "var ta=document.createElement('textarea');ta.value=t;"
        "ta.style.position='fixed';ta.style.opacity='0';"
        "document.body.appendChild(ta);ta.select();"
        "try{document.execCommand('copy');}catch(e){}"
        "ta.remove();}"
        "if(navigator.clipboard&&navigator.clipboard.writeText){"
        "navigator.clipboard.writeText(t).catch(fallback);}"
        "else{fallback();}})();"
    )
    return f"<script>{js}</script>"


def gtts_audio_bytes(text: str, lang: str = "en") -> bytes | None:
    """Synthesise speech server-side with gTTS; returns MP3 bytes."""
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def pyttsx3_speak(text: str) -> bool:
    """Offline TTS via pyttsx3 for CLI tools. Returns True on success."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception:
        return False
