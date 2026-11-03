"""
EmotionLens — Prediction overlay video

Renders a session's video next to what the classifier predicted at each moment:
the session on the left, and on the right the label the app showed, the model's
probability for every emotion, and a badge while a micro-expression was
detected. It is the side-by-side view used to show and check the analysis.

The source is the session's own video: the offline EVM render for a live
session (the raw recording is deleted after rendering), or the uploaded file
for a video upload. Live sessions need a time scale: the recorder writes at a
fixed 20 fps, so when the server analysed fewer frames per second the video is
shorter than the session, and record timestamps are mapped onto it in
proportion. Uploads are timestamped in video time already (scale 1).
"""

import bisect
import os
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.app.services.evm_renderer import _write_progress
from backend.app.utils.video_codec import ensure_browser_playable

WIDTH, HEIGHT, VIDEO_W = 1280, 720, 960

# Order, labels and colours as the dashboard shows them (config.js, styles.css).
EMOTIONS = [
    ("happy", "Feliz", (224, 174, 67)),
    ("neutral", "Neutral", (169, 166, 159)),
    ("angry", "Enojado", (206, 134, 134)),
    ("sad", "Triste", (128, 152, 190)),
    ("surprise", "Sorpresa", (179, 150, 206)),
    ("disgust", "Disgusto", (150, 184, 136)),
    ("nervousness", "Nervioso", (216, 154, 108)),
    ("confidence", "Confiado", (127, 175, 194)),
]
MICRO_LABELS = {
    "stress": "Tensión", "anger": "Enojo", "contempt": "Desprecio", "disgust": "Desagrado",
    "fear": "Miedo", "sadness": "Tristeza", "surprise": "Sorpresa",
}

# A record older than this is not "what the model sees now": low light, no
# face or a quality gate dropped the frames in between, so say so instead of
# freezing the last label on screen.
STALE_AFTER_S = 1.0
MICRO_BADGE_S = 2.0

BG, PANEL, TEXT, MUTED, TRACK = (18, 18, 20), (30, 30, 34), (236, 234, 228), (150, 148, 142), (52, 52, 58)

_FONT_FILES = {
    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"],
    True: ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"],
}


def _font(size: int, bold: bool = False):
    for path in _FONT_FILES[bold]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def output_paths(recordings_dir: str, session_id: int) -> tuple[str, str]:
    """Video and progress file. Both start with "<id>_", so deleting the
    session removes them with the rest of its recordings."""
    return (os.path.join(recordings_dir, f"{session_id}_prediction.mp4"),
            os.path.join(recordings_dir, f"{session_id}_prediction_progress.json"))


def record_at(timestamps: list[float], t: float) -> int | None:
    """Index of the last record at or before session time t, or None when
    there is none or it is stale (see STALE_AFTER_S)."""
    i = bisect.bisect_right(timestamps, t) - 1
    if i < 0 or t - timestamps[i] > STALE_AFTER_S:
        return None
    return i


class _Panel:
    def __init__(self):
        self.f_head = _font(20)
        self.f_small = _font(17)
        self.f_label = _font(40, bold=True)
        self.f_bar = _font(21)
        self.f_badge = _font(22, bold=True)

    def draw(self, d: ImageDraw.ImageDraw, session_id: int, clock_s: float,
             emotion: str | None, probs: dict | None) -> None:
        x, right = VIDEO_W + 26, WIDTH - 26
        d.rectangle([VIDEO_W, 0, WIDTH, HEIGHT], fill=PANEL)
        d.text((x, 22), "EMOTION-LENS", font=self.f_head, fill=MUTED)
        d.text((x, 48), f"Sesión {session_id}  ·  {int(clock_s) // 60}:{int(clock_s) % 60:02d}",
               font=self.f_small, fill=MUTED)
        d.text((x, 100), "Emoción detectada", font=self.f_small, fill=MUTED)

        if emotion is None:
            d.rounded_rectangle([x, 126, right, 196], radius=12, outline=MUTED, width=2)
            d.text((x + 16, 140), "Sin lectura", font=_font(30, bold=True), fill=MUTED)
            d.text((x, 206), "Poca luz o rostro no visible", font=self.f_small, fill=MUTED)
        else:
            label, color = next(((l, c) for k, l, c in EMOTIONS if k == emotion), (emotion, TEXT))
            d.rounded_rectangle([x, 126, right, 196], radius=12, fill=color)
            d.text((x + 16, 132), label, font=self.f_label, fill=(24, 24, 26))

        d.text((x, 240), "Probabilidades del modelo", font=self.f_small, fill=MUTED)
        y = 268
        for key, name, color in EMOTIONS:
            p = float((probs or {}).get(key, 0.0)) if emotion is not None else 0.0
            active = key == emotion
            d.text((x, y), name, font=self.f_bar, fill=TEXT if active else MUTED)
            d.text((right, y), f"{p * 100:.0f}%" if emotion is not None else "–",
                   font=self.f_bar, fill=TEXT if active else MUTED, anchor="ra")
            d.rounded_rectangle([x, y + 30, right, y + 38], radius=4, fill=TRACK)
            if p > 0.005:
                d.rounded_rectangle([x, y + 30, x + max(8, int((right - x) * p)), y + 38], radius=4, fill=color)
            y += 52

    def badge(self, d: ImageDraw.ImageDraw, micro: str) -> None:
        d.rounded_rectangle([20, 20, 420, 74], radius=10, fill=(20, 20, 22))
        d.text((36, 33), f"Microexpresión: {MICRO_LABELS.get(micro, micro)}", font=self.f_badge,
               fill=(240, 200, 120))


def render_prediction_video(
    session_id: int,
    source_video: str,
    records: list[dict],
    micro_expressions: list[dict],
    output_path: str,
    progress_path: str,
    session_duration: float | None = None,
) -> str | None:
    """
    Write the side-by-side video. `records` are emotion records (timestamp,
    emotion, emotion_probabilities) in session time; `session_duration` is set
    for live sessions to map that time onto the video, None for uploads.
    Returns output_path, or None on failure (the progress file says why).
    """
    _write_progress(progress_path, "rendering", 0.0, "initializing")
    tmp_path = output_path.replace(".mp4", ".part.mp4")
    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            raise RuntimeError(f"no se pudo abrir el video {source_video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_s = total / fps
        scale = (session_duration / video_s) if session_duration and video_s > 0 else 1.0

        records = sorted(records, key=lambda r: r["timestamp"])
        stamps = [r["timestamp"] for r in records]
        micros = sorted((m["timestamp"], m["detected_emotion"]) for m in micro_expressions)

        codec_used = None
        for codec in ["avc1", "H264", "mp4v"]:
            writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*codec), fps, (WIDTH, HEIGHT))
            if writer.isOpened():
                codec_used = codec
                break
            writer.release()
            writer = None
        if writer is None:
            raise RuntimeError("no se pudo abrir el escritor de video")

        panel = _Panel()
        cached_key, cached_panel = None, None
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_video = k / fps
            t_session = t_video * scale
            i = record_at(stamps, t_session)
            micro = next((name for ts, name in micros if 0 <= t_session - ts <= MICRO_BADGE_S), None)

            # The panel only changes with the record and the clock's second,
            # so it is drawn once per change and reused across frames.
            key = (i, int(t_video))
            if key != cached_key:
                full = Image.new("RGB", (WIDTH, HEIGHT), BG)
                rec = records[i] if i is not None else None
                panel.draw(ImageDraw.Draw(full), session_id, t_video,
                           rec["emotion"] if rec else None,
                           rec.get("emotion_probabilities") if rec else None)
                img = full.crop((VIDEO_W, 0, WIDTH, HEIGHT))
                cached_panel = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                cached_key = key

            out = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
            out[:, :VIDEO_W] = cv2.resize(frame, (VIDEO_W, HEIGHT), interpolation=cv2.INTER_LINEAR)
            out[:, VIDEO_W:] = cached_panel
            if micro:
                pil = Image.fromarray(cv2.cvtColor(out[:80, :440], cv2.COLOR_BGR2RGB))
                panel.badge(ImageDraw.Draw(pil), micro)
                out[:80, :440] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            writer.write(out)

            k += 1
            if k % 40 == 0:
                _write_progress(progress_path, "rendering", min(0.95, 0.95 * k / total), f"frames {k}/{total}")

        writer.release()
        writer = None
        if k == 0:
            raise RuntimeError("el video no tiene fotogramas")
        _write_progress(progress_path, "rendering", 0.96, "encoding")
        ensure_browser_playable(tmp_path, codec_used)
        os.replace(tmp_path, output_path)
        _write_progress(progress_path, "ready", 1.0, "done")
        print(f"[OK] Prediction video rendered: {output_path} ({k} frames)")
        return output_path
    except Exception as e:  # noqa: BLE001 — a failed render must only mark itself failed
        print(f"[ERR] Prediction video failed for session {session_id}: {e}")
        traceback.print_exc()
        _write_progress(progress_path, "failed", 0.0, str(e)[:200])
        return None
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
