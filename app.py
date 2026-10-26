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
                   page_icon=":vulcan_salute:", layout="wide")

QUALITY = {
    "Low (fastest - best for the cloud)": (480, 360, 15),
    "Medium": (640, 480, 24),
    "High": (960, 540, 30),
}


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


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## :vulcan_salute: SignAI")
    st.caption("Real-time sign language recognition prototype (ASL)")

    mode_label = st.radio(
        "Recognition mode",
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
    with st.expander("Detection settings", expanded=False):
        settings.min_score = st.slider("Confidence threshold", 0.30, 0.95, settings.min_score, 0.05)
        settings.stable_frames = st.slider("Stability (frames a sign must hold)", 2, 15, settings.stable_frames)
        settings.cooldown = st.slider("Cooldown between same words (s)", 0.5, 4.0, settings.cooldown, 0.25)
        settings.auto_append = st.checkbox("Auto-append recognised words to the sentence", True)
        st.checkbox("Auto-speak each recognised word", key="auto_speak_word")

    tts_engine = st.radio("Speech engine", ["Browser (Web Speech)", "gTTS (server audio)"],
                          index=0, horizontal=False)
    voice_rate = st.slider("Speech rate", 0.5, 2.0, 1.05, 0.05)
    st.checkbox("Live UI refresh", key="auto_refresh",
                help="Keep this ON so panels update while the camera runs.")

    quality = st.select_slider("Camera quality", list(QUALITY.keys()), list(QUALITY.keys())[0])

    with st.expander("Demo vocabulary", expanded=False):
        st.caption("Pretrained pose -> demo word (a prototype mapping, NOT full ASL):")
        st.dataframe(pd.DataFrame({
            "Hand pose": GESTURE_LABELS,
            "Demo word": [word_for_gesture(g) for g in GESTURE_LABELS],
        }), hide_index=True, width="stretch")

    st.divider()
    st.caption("ASL is a full language with grammar, movement and expression. "
               "This prototype recognises a limited sign set - see the About tab.")


# ------------------------------------------------------------------ main area
st.markdown("### Real-Time Sign Language Recognition")
st.caption("Show a hand pose to the camera. Hold it steady - recognised words build up "
           "into a sentence you can speak aloud. :warning: Demo mode maps hand *poses* to "
           "words; train your own model for real ASL vocabulary.")

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
        st.caption(":green[Camera running] - allow browser camera access if prompted.")
    else:
        st.caption("Click **START** above and allow camera access.")

with panel_col:
    st.subheader("Recognition")
    latest = (ctx.video_processor.latest if ctx and ctx.video_processor
              else {"word": "-", "label": "", "score": 0.0, "fps": 0.0,
                    "hands": 0, "backend": "-"})
    m1, m2, m3 = st.columns(3)
    m1.metric("Current sign", latest["word"])
    m2.metric("Confidence", f"{latest['score']:.0%}")
    m3.metric("Words signed", len(ss.history))
    st.progress(float(min(latest["score"], 1.0)))
    st.caption(f"Backend: `{latest['backend']}` | FPS: {latest['fps']:.0f} | Hands: {latest['hands']}")

# drain committed words AFTER widgets exist so counts/panel stay in sync
last_word = drain_words(ctx)

st.divider()
st.subheader("Sentence")
sent_col, btn_col = st.columns([3, 2])

with sent_col:
    st.text_area("Recognised sentence (editable)", key="sentence", height=120,
                 help="Type freely too - the sentence is yours to correct.")

with btn_col:
    b1, b2, b3 = st.columns(3)
    speak_clicked = b1.button(":loudspeaker: Speak", type="primary", width="stretch")
    if b2.button(":broom: Clear", width="stretch"):
        ss.sentence = ""
        ss.counts.clear()
    if b3.button(":leftwards_arrow_with_hook: Backspace", width="stretch"):
        parts = ss.sentence.strip().split(" ")
        ss.sentence = " ".join(parts[:-1])
    b4, b5, _ = st.columns(3)
    if b4.button("Space", width="stretch"):
        ss.sentence += " "
    copy_clicked = b5.button("Copy", width="stretch")

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
        st.toast("Sentence copied to clipboard")

if last_word and ss.get("auto_speak_word"):
    spoken = word_for_gesture(last_word) if settings.mode == "demo" else last_word
    if spoken:
        st.components.v1.html(browser_speak_html(spoken, rate=voice_rate), height=0)

st.divider()
hist_col, chart_col = st.columns([2, 1])
with hist_col:
    st.subheader("Sign history")
    if ss.history:
        st.dataframe(pd.DataFrame(ss.history[-20:][::-1]),
                     hide_index=True, width="stretch", height=300)
    else:
        st.caption("No signs recognised yet in this session.")
with chart_col:
    st.subheader("Word counts")
    if ss.counts:
        st.bar_chart(pd.Series(ss.counts, name="count"))
    else:
        st.caption("Statistics appear as you sign.")

# ----------------------------------------------------------------------- tabs
tab_use, tab_train, tab_about = st.tabs(
    ["How to use", "Train your own model", "About & disclaimer"])

with tab_use:
    st.markdown(
        """
        1. **START** the camera and allow browser access.
        2. Hold one of the demo hand poses about 40-60 cm from the lens, palm facing the camera.
        3. Keep it still until the word appears - it is appended to the sentence automatically.
        4. Pause (lower your hand) between signs, then do the next one.
        5. Edit the sentence if needed, then press **Speak**.

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
        **What this is** - an educational prototype: webcam -> MediaPipe hand
        landmarks -> (rule-based or LSTM) classifier -> words -> speech.

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

# ------------------------------------------------------------ live UI refresh
if playing and ss.auto_refresh:
    time.sleep(0.5)
    st.rerun()
