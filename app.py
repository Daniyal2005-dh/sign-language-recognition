"""SignAI - Real-Time Sign Language Recognition (Streamlit app).

Run locally:      streamlit run app.py
Deployed:         Streamlit Community Cloud picks this file up automatically.

The app streams webcam video with streamlit-webrtc, runs MediaPipe hand
landmarks + gesture recognition on every frame, converts held gestures into
words (with smoothing), builds a sentence, and speaks it.

Two recognition modes:
* "Built-in demo gestures" - works instantly (pretrained MediaPipe poses).
* "My trained model"       - loads an LSTM/Transformer trained with train.py.
"""
from __future__ import annotations

import html as _html
import queue
import sys
import time
from collections import Counter
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from signai.config import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    GESTURE_LABELS,
    MODELS_DIR,
    word_for_gesture,
)
from signai.features import frame_features  # noqa: E402
from signai.hands import HandPipeline, draw_hand_frame  # noqa: E402
from signai.predict import SequencePredictor, WordCommitter  # noqa: E402
from signai.speech import browser_speak_html, copy_to_clipboard_html, gtts_audio_bytes  # noqa: E402

import av  # noqa: E402
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer  # noqa: E402

st.set_page_config(page_title="SignAI - Sign Language Recognition",
                   page_icon="🖖", layout="wide")

QUALITY = {
    "Low (fastest - best for the cloud)": (480, 360, 15),
    "Medium": (640, 480, 24),
    "High": (960, 540, 30),
}

REPO_URL = "https://github.com/Daniyal2005-dh/sign-language-recognition"


# ---------------------------------------------------------------- theme & CSS
def inject_css() -> None:
    st.html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
          :root {
            --sig-bg: #0b0e14;
            --sig-card: #12172280;
            --sig-card-solid: #121722;
            --sig-border: #232b3d;
            --sig-violet: #8b7cff;
            --sig-cyan: #38e1d4;
            --sig-text: #eaecf5;
            --sig-dim: #9aa3b8;
          }
          html, body, [class*="css"], .stApp {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
          }
          .stApp { background:
            radial-gradient(1200px 600px at 85% -10%, rgba(139,124,255,.14), transparent 60%),
            radial-gradient(900px 500px at -10% 110%, rgba(56,225,212,.10), transparent 55%),
            var(--sig-bg) !important; }

          /* ---------- hero ---------- */
          .sig-hero {
            background: linear-gradient(135deg, rgba(139,124,255,.16), rgba(56,225,212,.08) 55%, rgba(139,124,255,.05)),
                        var(--sig-card-solid);
            border: 1px solid var(--sig-border);
            border-radius: 20px; padding: 26px 30px 22px; margin: 4px 0 18px;
            position: relative; overflow: hidden;
          }
          .sig-hero::after {
            content: ''; position: absolute; inset: -40%;
            background: conic-gradient(from 180deg at 50% 50%, transparent 0deg, rgba(139,124,255,.10) 90deg, transparent 180deg);
            animation: sig-spin 14s linear infinite; pointer-events: none;
          }
          @keyframes sig-spin { to { transform: rotate(1turn); } }
          .sig-hero h1 {
            margin: 0; font-weight: 800; letter-spacing: -.5px; line-height: 1.1;
            font-size: clamp(26px, 3.4vw, 40px);
            background: linear-gradient(90deg, #ffffff 10%, #b9aaff 55%, #67e8dc);
            -webkit-background-clip: text; background-clip: text; color: transparent;
          }
          .sig-hero p { color: var(--sig-dim); margin: 8px 0 0; max-width: 760px; font-size: .95rem; }
          .sig-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
          .sig-badge {
            font-size: .74rem; font-weight: 600; letter-spacing: .2px;
            padding: 5px 12px; border-radius: 999px;
            border: 1px solid var(--sig-border); background: rgba(139,124,255,.10);
            color: #c9c2ff; white-space: nowrap;
          }
          .sig-badge.cyan { background: rgba(56,225,212,.08); color: #7deee2; }

          /* ---------- cards & panels ---------- */
          .sig-card {
            background: var(--sig-card); border: 1px solid var(--sig-border);
            border-radius: 16px; padding: 18px 20px;
          }
          .sig-label { color: var(--sig-dim); font-size: .78rem; font-weight: 600;
                       text-transform: uppercase; letter-spacing: 1.2px; }

          /* ---------- live word card ---------- */
          .sig-word {
            font-size: clamp(38px, 5vw, 64px); font-weight: 800; line-height: 1.05;
            letter-spacing: -1px; margin: 10px 0 6px; min-height: 1.2em;
            background: linear-gradient(90deg, #ffffff, #b9aaff);
            -webkit-background-clip: text; background-clip: text; color: transparent;
          }
          .sig-conf-track { height: 8px; border-radius: 999px; background: #1c2333; overflow: hidden; }
          .sig-conf-fill { height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, var(--sig-violet), var(--sig-cyan));
            transition: width .3s ease; }
          .sig-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
          .sig-chip {
            font-size: .74rem; color: var(--sig-dim); border: 1px solid var(--sig-border);
            background: #0f1420; border-radius: 999px; padding: 4px 11px; white-space: nowrap;
          }
          .sig-chip b { color: var(--sig-text); font-weight: 600; }
          .sig-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                     margin-right: 6px; background: #3ddc84; box-shadow: 0 0 8px #3ddc84; }
          .sig-dot.off { background: #5b6478; box-shadow: none; }

          /* ---------- sentence ---------- */
          .sig-sentence {
            font-size: clamp(20px, 2.4vw, 30px); font-weight: 700; line-height: 1.35;
            min-height: 46px; color: #f4f5fb;
          }
          .sig-sentence .cursor { display: inline-block; width: 3px; height: 1.05em;
            background: var(--sig-cyan); vertical-align: -0.15em; margin-left: 4px;
            animation: sig-blink 1.1s steps(1) infinite; }
          @keyframes sig-blink { 50% { opacity: 0; } }
          .sig-sentence-empty { color: #5b6478; font-weight: 500; font-style: italic; }

          /* ---------- word chips (history) ---------- */
          .sig-words { display: flex; flex-wrap: wrap; gap: 7px; }
          .sig-w {
            animation: sig-pop .25s ease; font-size: .82rem; font-weight: 600;
            padding: 5px 12px; border-radius: 999px; color: #dcd9ff;
            background: rgba(139,124,255,.14); border: 1px solid rgba(139,124,255,.35);
          }
          .sig-w span { color: var(--sig-dim); font-weight: 500; font-size: .7rem; margin-left: 6px; }
          @keyframes sig-pop { from { transform: scale(.7); opacity: 0; } to { transform: scale(1); opacity: 1; } }

          /* ---------- streamlit widget restyle ---------- */
          .stButton > button {
            border-radius: 12px !important; font-weight: 600 !important;
            border: 1px solid var(--sig-border) !important;
            background: #131a28 !important; color: var(--sig-text) !important;
            transition: transform .08s ease, box-shadow .15s ease, border-color .15s ease;
          }
          .stButton > button:hover { border-color: #3a4460 !important; box-shadow: 0 4px 18px rgba(139,124,255,.15); }
          .stButton > button:active { transform: scale(.97); }
          .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7c5cff, #5f8bff) !important; border: none !important;
            box-shadow: 0 6px 20px rgba(124,92,255,.35);
          }
          .stButton > button[kind="primary"]:hover { box-shadow: 0 8px 26px rgba(124,92,255,.5); }
          div[data-testid="stTextAreaTextarea"] {
            border-radius: 12px !important; background: #10151f !important;
          }
          .stTabs [data-baseweb="tab-list"] { gap: 6px; }
          .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0; padding: 8px 16px;
            color: var(--sig-dim) !important; font-weight: 600;
          }
          .stTabs [aria-selected="true"] { color: #c9c2ff !important; }
          .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
          .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--sig-border); }
          div[data-testid="stExpander"] {
            border: 1px solid var(--sig-border) !important; border-radius: 12px !important;
            background: rgba(18,23,34,.6);
          }
          .stSidebar { border-right: 1px solid var(--sig-border); }
          .stSidebar .side-logo {
            font-weight: 800; font-size: 1.35rem; letter-spacing: -.3px;
            background: linear-gradient(90deg, #ffffff, #b9aaff 60%, #67e8dc);
            -webkit-background-clip: text; background-clip: text; color: transparent;
          }
          .stSidebar hr { border-color: var(--sig-border) !important; }

          /* ---------- footer ---------- */
          .sig-footer { color: #6b7387; font-size: .8rem; text-align: center; padding: 18px 0 6px; }
          .sig-footer a { color: #9a8fff; text-decoration: none; font-weight: 600; }

          /* compact number inputs / sliders spacing */
          .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px; }
        </style>
        """,
    )


def hero() -> None:
    st.markdown(
        f"""
        <div class="sig-hero">
          <h1>🖖 SignAI &mdash; Sign Language → Speech</h1>
          <p>Show a hand pose to the camera and hold it steady. Recognised signs build into a
          sentence you can speak aloud &mdash; in real time, entirely in your browser.
          <b>Demo mode maps hand poses to words; train your own model for real ASL vocabulary.</b></p>
          <div class="sig-badges">
            <span class="sig-badge">🖐 21-landmark hand tracking</span>
            <span class="sig-badge cyan">⚡ Real-time LSTM / Transformer</span>
            <span class="sig-badge">🔊 Speech output</span>
            <span class="sig-badge cyan">🔒 Nothing recorded</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def live_word_card(latest: dict, playing: bool) -> None:
    word = _html.escape(str(latest.get("word", "-")))
    score = float(min(latest.get("score", 0.0), 1.0))
    fps = latest.get("fps", 0.0)
    hands = latest.get("hands", 0)
    backend = _html.escape(str(latest.get("backend", "-")))
    st.markdown(
        f"""
        <div class="sig-card">
          <div class="sig-label">Current sign</div>
          <div class="sig-word">{_html.escape(word if word != "-" else "…")}</div>
          <div class="sig-label" style="margin-bottom:6px">Confidence &middot; {score:.0%}</div>
          <div class="sig-conf-track"><div class="sig-conf-fill" style="width:{max(score * 100, 2):.0f}%"></div></div>
          <div class="sig-chips">
            <span class="sig-chip"><span class="sig-dot{' off' if not playing else ''}"></span>{'Live' if playing else 'Camera off'}</span>
            <span class="sig-chip">⚙️ <b>{backend}</b></span>
            <span class="sig-chip">🎞 <b>{fps:.0f}</b> fps</span>
            <span class="sig-chip">🖐 <b>{hands}</b> hand{'s' if hands != 1 else ''}</span>
            <span class="sig-chip">📝 <b>{len(ss.history)}</b> words signed</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def word_chips() -> str:
    if not ss.history:
        return ""
    chips = []
    for h in ss.history[-14:][::-1]:
        chips.append(f'<span class="sig-w">{_html.escape(str(h["word"]))}'
                     f'<span>{_html.escape(str(h["time"]))}</span></span>')
    return f'<div class="sig-words">{"".join(chips)}</div>'


# --------------------------------------------------------------------- state
class Settings:
    """Mutable settings shared with the video-processing thread (GIL-safe)."""

    def __init__(self) -> None:
        self.min_score = 0.60
        self.stable_frames = 6
        self.cooldown = 1.5
        self.auto_append = True
        self.mode = "demo"            # "demo" | "custom"
        self.predictor = None         # SequencePredictor | None


ss = st.session_state
ss.setdefault("settings", Settings())
ss.setdefault("sentence", "")
ss.setdefault("history", [])          # [{time, word, confidence}]
ss.setdefault("counts", Counter())
ss.setdefault("auto_refresh", True)
ss.setdefault("auto_speak_word", False)
settings: Settings = ss.settings


def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


# ------------------------------------------------------------------ processor
class SignProcessor(VideoProcessorBase):
    """Runs on the webrtc frame thread: detect -> classify -> commit words."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.pipeline = HandPipeline(num_hands=2, use_task_model=True)
        self.committer = WordCommitter(stable_frames=cfg.stable_frames,
                                       cooldown_s=cfg.cooldown,
                                       min_score=cfg.min_score)
        self.word_queue: queue.Queue[str] = queue.Queue()
        self.latest = {"word": "-", "label": "", "score": 0.0,
                       "fps": 0.0, "hands": 0, "backend": self.pipeline.backend}
        self._fps = 0.0
        self._t_prev = time.time()
        self._buf: deque = deque(maxlen=DEFAULT_SEQ_LEN)
        self._seq_pred: SequencePredictor | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)                       # mirror "selfie" view
        hf = self.pipeline.process_bgr(img)

        cfg = self.cfg
        self.committer.min_score = cfg.min_score
        self.committer.stable_frames = cfg.stable_frames
        self.committer.cooldown_s = cfg.cooldown

        label, score = "None", 0.0
        if cfg.mode == "custom" and cfg.predictor is not None:
            if self._seq_pred is not cfg.predictor:      # new model loaded
                self._seq_pred = cfg.predictor
                self._buf = deque(maxlen=self._seq_pred.seq_len)
            feats = frame_features([h.landmarks for h in hf.hands])
            label, score = self._seq_pred.add_frame(feats)
            if label == "":
                label, score = "None", 0.0               # window still filling
        else:
            hand = hf.best_hand()
            if hand is not None:
                label, score = hand.gesture, hand.score

        word = self.committer.update(label, score)
        if word and cfg.auto_append:
            self.word_queue.put(word)

        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._t_prev, 1e-6)
        self._t_prev = now

        if cfg.mode == "demo":
            shown = word_for_gesture(label)
        else:
            shown = label
        if not shown:
            shown = "-"
        if label != "None":                          # hide banner when no hand is visible
            img = draw_hand_frame(img, hf, text=shown, conf=score, fps=self._fps)
        else:
            img = draw_hand_frame(img, hf, fps=self._fps)

        self.latest = {
            "word": shown,
            "label": label,
            "score": float(score),
            "fps": self._fps,
            "hands": len(hf.hands),
            "backend": self.pipeline.backend,
        }
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------------------------------------------- helpers
def append_word(raw: str, score: float) -> None:
    word = raw.strip()
    if not word:
        return
    if settings.mode == "demo":
        word = word_for_gesture(word) or word
    if ss.sentence and not ss.sentence.endswith(" "):
        ss.sentence += " "
    ss.sentence += word
    ss.counts[word] += 1
    ss.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "word": word,
        "confidence": round(float(score), 2),
    })


def drain_words(ctx) -> str | None:
    """Move committed words from the processor thread into the UI. Returns last word."""
    last = None
    if ctx and ctx.video_processor is not None:
        while True:
            try:
                raw = ctx.video_processor.word_queue.get_nowait()
            except queue.Empty:
                break
            append_word(raw, ctx.video_processor.latest.get("score", 0.0))
            last = raw
    return last


def load_custom_model(model_path: Path, labels_path: Path) -> SequencePredictor:
    if not tf_available():
        raise RuntimeError(
            "TensorFlow is not installed. Custom models need it: "
            "`pip install -r requirements-train.txt` (local only - keeps the cloud app light)."
        )
    return SequencePredictor(model_path, labels_path)


# ------------------------------------------------------------------------ app
inject_css()
hero()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="side-logo">🖖 SignAI</div>'
                '<div style="color:#9aa3b8;font-size:.85rem;margin:2px 0 10px">'
                'Real-time sign language recognition (ASL)</div>', unsafe_allow_html=True)

    mode_label = st.radio(
        "🎯 Recognition mode",
        ["Built-in demo gestures", "My trained model"],
        index=0,
        help="Demo = pretrained MediaPipe hand poses mapped to words. "
             "Trained = an LSTM/Transformer you trained with train.py.",
    )
    settings.mode = "custom" if mode_label.startswith("My") else "demo"

    if settings.mode == "custom":
        if not tf_available():
            st.warning("TensorFlow is not installed in this environment, so custom "
                       "models can't run here. Run locally with "
                       "`pip install -r requirements-train.txt`, or use the demo mode.")
        saved_models = sorted(p.stem for p in MODELS_DIR.glob("*.keras"))
        sel = st.selectbox("Saved models (models/)", ["- none -"] + saved_models)
        up_model = st.file_uploader("...or upload a .keras model", type=["keras"])
        up_labels = st.file_uploader("...with its labels .json", type=["json"])
        if st.button("Load model", type="primary", width="stretch"):
            try:
                if up_model is not None and up_labels is not None:
                    tmp = Path("data") / "uploads"
                    tmp.mkdir(parents=True, exist_ok=True)
                    mp_ = tmp / up_model.name
                    lp_ = tmp / up_labels.name
                    mp_.write_bytes(up_model.getvalue())
                    lp_.write_bytes(up_labels.getvalue())
                elif sel != "- none -":
                    mp_ = MODELS_DIR / f"{sel}.keras"
                    lp_ = MODELS_DIR / f"{sel}.labels.json"
                else:
                    raise RuntimeError("Pick a saved model or upload both files.")
                settings.predictor = load_custom_model(mp_, lp_)
                st.success(f"Loaded {mp_.stem} - classes: {', '.join(settings.predictor.labels)}")
            except Exception as exc:
                st.error(f"Could not load model: {exc}")
        if settings.predictor is None and settings.mode == "custom":
            st.info("No custom model loaded yet - the demo classifier will be used.")

    st.divider()
    with st.expander("🎛️ Detection settings", expanded=False):
        settings.min_score = st.slider("Confidence threshold", 0.30, 0.95, settings.min_score, 0.05)
        settings.stable_frames = st.slider("Stability (frames a sign must hold)", 2, 15, settings.stable_frames)
        settings.cooldown = st.slider("Cooldown between same words (s)", 0.5, 4.0, settings.cooldown, 0.25)
        settings.auto_append = st.checkbox("Auto-append recognised words to the sentence", True)
        st.checkbox("Auto-speak each recognised word", key="auto_speak_word")

    tts_engine = st.radio("🔊 Speech engine", ["Browser (Web Speech)", "gTTS (server audio)"],
                          index=0, horizontal=False)
    voice_rate = st.slider("Speech rate", 0.5, 2.0, 1.05, 0.05)
    st.checkbox("Live UI refresh", key="auto_refresh",
                help="Keep this ON so panels update while the camera runs.")

    quality = st.select_slider("📷 Camera quality", list(QUALITY.keys()), list(QUALITY.keys())[0])

    with st.expander("🤟 Demo vocabulary", expanded=False):
        st.caption("Pretrained pose → demo word (a prototype mapping, NOT full ASL):")
        st.dataframe(pd.DataFrame({
            "Hand pose": GESTURE_LABELS,
            "Demo word": [word_for_gesture(g) for g in GESTURE_LABELS],
        }), hide_index=True, width="stretch")

    st.divider()
    st.caption("ASL is a full language with grammar, movement and expression. "
               "This prototype recognises a limited sign set - see the About tab.")

# ------------------------------------------------------------------ main area
cam_col, panel_col = st.columns([3, 2], gap="large")

with cam_col:
    w, h, fr = QUALITY[quality]
    ctx = webrtc_streamer(
        key=f"signai-{quality}-{settings.mode}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: SignProcessor(settings),
        media_stream_constraints={
            "video": {"width": {"ideal": w}, "height": {"ideal": h},
                      "frameRate": {"ideal": fr}},
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    playing = bool(ctx and ctx.state.playing)
    if playing:
        st.success("🟢 Camera running - hold a pose steady to sign!")
    else:
        st.info("👆 Click **START** above, then allow camera access to begin signing.")

with panel_col:
    latest = (ctx.video_processor.latest if ctx and ctx.video_processor
              else {"word": "-", "label": "", "score": 0.0, "fps": 0.0,
                    "hands": 0, "backend": "-"})
    live_word_card(latest, playing)

# drain committed words AFTER widgets exist so counts/panel stay in sync
last_word = drain_words(ctx)

# ------------------------------------------------------------ sentence studio
st.markdown('<div class="sig-label" style="margin:20px 0 4px">Sentence studio</div>',
            unsafe_allow_html=True)
sentence = ss.sentence.strip()
if sentence:
    st.markdown(
        f'<div class="sig-card"><div class="sig-sentence">'
        f'{_html.escape(sentence)}<span class="cursor"></span></div></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="sig-card"><div class="sig-sentence sig-sentence-empty">'
        '👋 Sign something to begin - or just type here…</div></div>',
        unsafe_allow_html=True,
    )

sent_col, btn_col = st.columns([3, 2])

with sent_col:
    st.text_area("Recognised sentence (editable)", key="sentence", height=100,
                 help="Type freely too - the sentence is yours to correct.",
                 label_visibility="collapsed")

with btn_col:
    b1, b2, b3, b4, b5 = st.columns(5)
    speak_clicked = b1.button("🔊", help="Speak the sentence", type="primary", width="stretch")
    if b2.button("🗑️", help="Clear the sentence"):
        ss.sentence = ""
        ss.counts.clear()
    if b3.button("⌫", help="Backspace (remove last word)"):
        parts = ss.sentence.strip().split(" ")
        ss.sentence = " ".join(parts[:-1])
    if b4.button("␣", help="Insert a space"):
        ss.sentence += " "
    copy_clicked = b5.button("📋", help="Copy to clipboard")

    text = ss.sentence.strip()
    if speak_clicked:
        if not text:
            st.warning("Nothing to speak yet - sign something or type below.")
        elif tts_engine.startswith("Browser"):
            st.components.v1.html(browser_speak_html(text, rate=voice_rate), height=0)
        else:
            audio = gtts_audio_bytes(text)
            if audio:
                st.audio(audio, format="audio/mp3")
            else:
                st.error("gTTS failed (needs internet). Try the Browser engine.")

    if copy_clicked and text:
        st.components.v1.html(copy_to_clipboard_html(text), height=0)
        st.toast("📋 Sentence copied to clipboard")

if last_word and ss.get("auto_speak_word"):
    spoken = word_for_gesture(last_word) if settings.mode == "demo" else last_word
    if spoken:
        st.components.v1.html(browser_speak_html(spoken, rate=voice_rate), height=0)

if last_word:
    spoken = word_for_gesture(last_word) if settings.mode == "demo" else last_word
    st.toast(f"🤟 Signed: **{spoken}** - added to your sentence")

# ------------------------------------------------------------------- history
st.markdown('<div class="sig-label" style="margin:18px 0 4px">Session history</div>',
            unsafe_allow_html=True)
if ss.history:
    st.markdown(word_chips(), unsafe_allow_html=True)
    chip_rows = st.columns([2, 1])
    with chip_rows[0]:
        st.dataframe(pd.DataFrame(ss.history[-20:][::-1]),
                     hide_index=True, width="stretch", height=240)
    with chip_rows[1]:
        st.caption("Most signed words")
        st.bar_chart(pd.Series(ss.counts, name="count"), height=240)
else:
    st.caption("No signs recognised yet in this session - your word chips and stats appear here.")

# ----------------------------------------------------------------------- tabs
tab_use, tab_train, tab_about = st.tabs(
    ["❓ How to use", "🏋️ Train your own model", "ℹ️ About & disclaimer"])

with tab_use:
    st.markdown(
        """
        1. **START** the camera and allow browser access.
        2. Hold one of the demo hand poses about 40-60 cm from the lens, palm facing the camera.
        3. Keep it still until the word appears - it is appended to the sentence automatically.
        4. Pause (lower your hand) between signs, then do the next one.
        5. Edit the sentence if needed, then press **🔊 Speak**.

        | Hand pose | Demo word |
        |---|---|
        | Open palm | Hello |
        | Thumb + index + pinky | I Love You |
        | Index + middle (V) | Peace |
        | Thumb up | Good |
        | Thumb down | Bad |
        | Index pointing up | You |
        | Closed fist | Stop |
        """
    )

with tab_train:
    st.markdown(
        """
        Teach the app **your** vocabulary (real ASL words like *thank you*, *hello*, *please*).

        **1. Collect data** - one class at a time, several people if possible:

        ```bash
        pip install -r requirements-train.txt   # adds TensorFlow + scikit-learn
        python collect_data.py --label hello --samples 30
        python collect_data.py --label thankyou --samples 30
        python collect_data.py --label thankyou --samples 10 --signer friend2
        ```

        **2. Train and evaluate** (LSTM over 30-frame landmark sequences):

        ```bash
        python train.py --group-by-signer
        ```

        See `reports/` for the confusion matrix and metrics.

        **3. Use it here**: sidebar -> *My trained model* -> Load model
        (models must exist in `models/`; running custom models in this browser
        app requires TensorFlow installed in the serving environment).

        Tips:
        - Record **movement** for dynamic signs - the whole 1-second window matters.
        - Same signs, multiple angles/distances -> better generalisation.
        - Evaluate on a **different person** (`--group-by-signer`) to measure real-world accuracy.
        """
    )

with tab_about:
    st.markdown(
        """
        **What this is** - an educational prototype: webcam → MediaPipe hand
        landmarks → (rule-based or LSTM) classifier → words → speech.

        **What this is NOT** - a sign-language translator. Sign languages are
        full languages: meaning comes from handshape, movement, facial
        expression, body posture and context. A handful of poses or
        one-second clips cannot capture that, and ASL is not universal
        (e.g. Pakistan Sign Language differs).

        **Tech stack**: Python, OpenCV, MediaPipe, TensorFlow/Keras (LSTM +
        Transformer), scikit-learn, Streamlit, streamlit-webrtc, gTTS/Web Speech.

        **Privacy**: video frames are processed live; nothing is recorded or uploaded.

        **Roadmap ideas**: two-handed and dynamic ASL vocabulary, signer-independent
        evaluation, facial-expression features, PSL vocabulary, WLASL benchmarking.
        """
    )

st.markdown(
    f'<div class="sig-footer">Built with 💜 by <a href="{REPO_URL}" target="_blank">'
    f'Daniyal2005-dh</a> · <a href="{REPO_URL}" target="_blank">GitHub</a> · MIT License · '
    'runs 100% in your browser - no video is stored</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ live UI refresh
if playing and ss.auto_refresh:
    time.sleep(0.5)
    st.rerun()
