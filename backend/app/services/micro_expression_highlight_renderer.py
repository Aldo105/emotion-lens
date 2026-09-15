"""
EmotionLens — Micro-Expression Highlight Renderer

Post-processes a raw session video to produce a downloadable MP4 containing
short, Eulerian-amplified, slow-motion clips of every detected micro-expression
— so a human reviewer can visually validate what the Action Unit engine flagged.

This gives EVM a real, functional role in the micro-expression pipeline (visual
validation) rather than only visualizing heart rate. It reuses the same linear
Eulerian amplification primitives as `evm_renderer.py`, but tuned differently:

  - Higher temporal frequency band (2-12 Hz vs 0.8-2 Hz for pulse), matching the
    40-500ms duration of a micro-expression instead of a heartbeat cycle.
  - Shallower pyramid (2 levels vs 4) to preserve fine spatial detail (brows,
    lip corners, eyelids).
  - Amplifies the Y (luminance) channel too, not just I/Q chrominance — this
    exaggerates edge/gradient shifts caused by subtle motion, not just skin
    color change.
  - Each clip is slowed down by duplicating frames, since a 2-10 frame event at
    20 FPS is imperceptible at normal speed even when amplified.

Runs OFFLINE after the session ends, using the same raw video + bbox metadata
already recorded for the pulse EVM renderer.
"""

import os
import traceback

import cv2
import numpy as np

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from backend.app.services.evm_renderer import (
    _bgr_to_yiq,
    _yiq_to_bgr,
    _build_pyramid_down,
    _upsample,
    _create_feather_mask,
    _write_progress,
)

MIN_CLIP_FRAMES = 8
MIN_VALID_BBOX_RATIO = 0.5


class MicroExpressionHighlightRenderer:
    """
    Offline renderer that builds a single MP4 of amplified, slow-motion clips
    for every (or the top-N) detected micro-expression of a session.
    """

    def __init__(
        self,
        amplification: float = 12.0,
        freq_low: float = 2.0,
        freq_high: float = 12.0,
        pyramid_levels: int = 2,
        clip_padding_sec: float = 0.75,
        max_clips: int = 15,
        slowmo_factor: int = 4,
    ):
        self.amplification = amplification
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.pyramid_levels = pyramid_levels
        self.clip_padding_sec = clip_padding_sec
        self.max_clips = max_clips
        self.slowmo_factor = max(1, slowmo_factor)

    def render(
        self,
        raw_video_path: str,
        metadata_path: str,
        events: list[dict],
        output_path: str,
    ) -> str | None:
        """
        Render the micro-expression highlight video.

        Args:
            raw_video_path: Path to the raw session video (MP4).
            metadata_path: Path to the .npz metadata (face bboxes, fps, etc.).
            events: MicroExpression rows as plain dicts (timestamp, duration_ms,
                detected_emotion, dominant_emotion_at_time, action_units_involved,
                relevance_score, is_contradictory, description).
            output_path: Path for the output highlight MP4.

        Returns:
            output_path on success, None if no video could be produced
            (e.g. no events, or every candidate clip was invalid).
        """
        if not SCIPY_AVAILABLE:
            print("[ERR] scipy required for micro-expression rendering but not installed")
            return None

        progress_path = output_path.replace(".mp4", "_progress.json")

        if not events:
            _write_progress(progress_path, "not_available", 0.0, "no micro-expressions")
            return None

        _write_progress(progress_path, "rendering", 0.0, "initializing")

        try:
            result = self._render_impl(
                raw_video_path, metadata_path, events, output_path, progress_path
            )
            _write_progress(
                progress_path,
                "ready" if result else "failed",
                1.0 if result else 0.0,
                "done" if result else "error",
            )
            return result
        except Exception as e:
            print(f"[ERR] Micro-expression highlight rendering failed: {e}")
            traceback.print_exc()
            _write_progress(progress_path, "failed", 0.0, str(e))
            return None

    def _render_impl(
        self,
        raw_video_path: str,
        metadata_path: str,
        events: list[dict],
        output_path: str,
        progress_path: str,
    ) -> str | None:
        meta = np.load(metadata_path, allow_pickle=True)
        face_bboxes = meta["face_bboxes"]  # (N, 4) int32
        fps_arr = meta["fps"]
        fps = float(fps_arr[0]) if len(fps_arr) > 0 else 20.0

        cap = cv2.VideoCapture(raw_video_path)
        if not cap.isOpened():
            print(f"[ERR] Cannot open raw video: {raw_video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames < MIN_CLIP_FRAMES:
            print(f"[WARN] Video too short for micro-expression rendering ({total_frames} frames)")
            cap.release()
            return None

        # Top-N by relevance, then chronological for the final montage
        selected = sorted(events, key=lambda e: e.get("relevance_score", 0), reverse=True)
        selected = selected[: self.max_clips]
        selected.sort(key=lambda e: e.get("timestamp", 0.0))

        writer = None
        for codec in ["avc1", "H264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
            if writer.isOpened():
                break
            writer.release()
            writer = None

        if writer is None:
            print("[ERR] Cannot open VideoWriter for micro-expression highlight output")
            cap.release()
            return None

        rendered_count = 0
        for idx, event in enumerate(selected):
            _write_progress(
                progress_path, "rendering", idx / max(len(selected), 1),
                f"clip {idx + 1}/{len(selected)}"
            )
            ok = self._render_one_clip(
                cap, face_bboxes, fps, total_frames, frame_w, frame_h,
                event, idx, len(selected), writer,
            )
            if ok:
                rendered_count += 1

        writer.release()
        cap.release()

        if rendered_count == 0:
            print("[WARN] No valid micro-expression clips could be rendered")
            try:
                os.remove(output_path)
            except Exception:
                pass
            return None

        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"[OK] Micro-expression highlight video rendered: {output_path} "
            f"({rendered_count}/{len(selected)} clips, {output_size:.1f} MB)"
        )
        return output_path

    def _render_one_clip(
        self,
        cap: cv2.VideoCapture,
        face_bboxes: np.ndarray,
        fps: float,
        total_frames: int,
        frame_w: int,
        frame_h: int,
        event: dict,
        clip_index: int,
        clip_total: int,
        writer: cv2.VideoWriter,
    ) -> bool:
        timestamp = float(event.get("timestamp", 0.0))
        center_frame = int(round(timestamp * fps))
        pad_frames = int(round(self.clip_padding_sec * fps))

        start_frame = max(0, center_frame - pad_frames)
        end_frame = min(total_frames, center_frame + pad_frames)

        if end_frame - start_frame < MIN_CLIP_FRAMES:
            print(f"[WARN] Skipping micro-expression at {timestamp:.2f}s — clip too short")
            return False

        clip_bboxes = face_bboxes[start_frame:end_frame]
        valid_mask = clip_bboxes[:, 0] >= 0
        if valid_mask.mean() < MIN_VALID_BBOX_RATIO:
            print(f"[WARN] Skipping micro-expression at {timestamp:.2f}s — face lost in clip")
            return False

        median_bbox = np.median(clip_bboxes[valid_mask], axis=0).astype(int)
        bw = median_bbox[2] - median_bbox[0]
        bh = median_bbox[3] - median_bbox[1]
        expand_x = int(bw * 0.15)
        expand_y = int(bh * 0.15)

        x1 = max(0, median_bbox[0] - expand_x)
        y1 = max(0, median_bbox[1] - expand_y)
        x2 = min(frame_w, median_bbox[2] + expand_x)
        y2 = min(frame_h, median_bbox[3] + expand_y)
        roi_h, roi_w = y2 - y1, x2 - x1

        if roi_h < 20 or roi_w < 20:
            print(f"[WARN] Skipping micro-expression at {timestamp:.2f}s — face ROI too small")
            return False

        # ── Read clip frames ────────────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip_frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            clip_frames.append(frame)

        n_frames = len(clip_frames)
        if n_frames < MIN_CLIP_FRAMES:
            return False

        # ── Build YIQ pyramid stack for this clip ───────────────────
        test_down = _build_pyramid_down(
            np.zeros((roi_h, roi_w, 3), dtype=np.float32), self.pyramid_levels
        )
        pyr_h, pyr_w = test_down.shape[:2]
        pyramid_stack = np.zeros((n_frames, pyr_h, pyr_w, 3), dtype=np.float32)

        for i, frame in enumerate(clip_frames):
            roi = frame[y1:y2, x1:x2].astype(np.float32) / 255.0
            yiq = _bgr_to_yiq(roi)
            level = _build_pyramid_down(yiq, self.pyramid_levels)
            ph = min(level.shape[0], pyr_h)
            pw = min(level.shape[1], pyr_w)
            pyramid_stack[i, :ph, :pw, :] = level[:ph, :pw, :]

        # ── Temporal filtering (all 3 channels: Y, I, Q) ────────────
        nyquist = fps / 2.0
        low = self.freq_low / nyquist
        high = min(self.freq_high / nyquist, 0.95)

        filtered_stack = np.zeros_like(pyramid_stack)
        if low < high and low > 0:
            order = 3 if n_frames > 20 else 2
            try:
                b, a = butter(order, [low, high], btype="band")
                padlen = min(3 * max(len(a), len(b)), n_frames - 1)
                if padlen >= 1:
                    for ch in range(3):
                        filtered_stack[:, :, :, ch] = filtfilt(
                            b, a, pyramid_stack[:, :, :, ch], axis=0, padlen=padlen
                        )
                else:
                    filtered_stack = self._high_pass_fallback(pyramid_stack)
            except Exception as e:
                print(f"[WARN] Bandpass filter failed for clip at {timestamp:.2f}s ({e}), using fallback")
                filtered_stack = self._high_pass_fallback(pyramid_stack)
        else:
            filtered_stack = self._high_pass_fallback(pyramid_stack)

        filtered_stack *= self.amplification

        # ── Reconstruct amplified frames ────────────────────────────
        feather_mask = _create_feather_mask(roi_h, roi_w, border=10)
        amplified_frames = []
        for i, frame in enumerate(clip_frames):
            upsampled = _upsample(filtered_stack[i], self.pyramid_levels, roi_h, roi_w)
            roi_bgr = frame[y1:y2, x1:x2].astype(np.float32) / 255.0
            roi_yiq = _bgr_to_yiq(roi_bgr)
            roi_yiq += upsampled
            roi_magnified = _yiq_to_bgr(roi_yiq)
            roi_magnified = np.clip(roi_magnified * 255.0, 0, 255).astype(np.uint8)

            out_frame = frame.copy()
            frame_f = frame[y1:y2, x1:x2].astype(np.float32)
            magnified_f = roi_magnified.astype(np.float32)
            blended = magnified_f * feather_mask + frame_f * (1.0 - feather_mask)
            out_frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            amplified_frames.append(out_frame)

        # ── Title card ───────────────────────────────────────────────
        self._write_title_card(writer, frame_w, frame_h, event, clip_index, clip_total, fps=fps)

        # ── Slow motion + annotation overlay, write to output ───────
        session_time_str = self._format_mmss(timestamp)
        for frame in amplified_frames:
            annotated = frame.copy()
            self._draw_annotation(annotated, frame_w, frame_h, event, session_time_str)
            for _ in range(self.slowmo_factor):
                writer.write(annotated)

        return True

    def _high_pass_fallback(self, pyramid_stack: np.ndarray) -> np.ndarray:
        """Simple high-pass via subtraction of the temporal mean, used when
        the clip is too short for a stable Butterworth filtfilt."""
        mean = pyramid_stack.mean(axis=0, keepdims=True)
        return pyramid_stack - mean

    def _format_mmss(self, seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _draw_annotation(
        self,
        frame: np.ndarray,
        frame_w: int,
        frame_h: int,
        event: dict,
        session_time_str: str,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        emotion = event.get("detected_emotion", "?")
        dominant = event.get("dominant_emotion_at_time", "?")
        relevance = event.get("relevance_score", 0)
        duration_ms = event.get("duration_ms", 0.0)
        aus = event.get("action_units_involved", [])
        is_contradictory = event.get("is_contradictory", False)

        lines = [
            f"Microexpresion: {emotion}  (dominante: {dominant})",
            f"AUs: {'+'.join(aus)}   Duracion: {duration_ms:.0f}ms   Relevancia: {relevance}/100",
            f"T={session_time_str}   SLOW-MO {self.slowmo_factor}x",
        ]
        if is_contradictory:
            lines.insert(0, "CONTRADICE LA EMOCION DOMINANTE")

        y = frame_h - 15 - (len(lines) - 1) * 22
        for line in lines:
            (tw, th), baseline = cv2.getTextSize(line, font, 0.55, 1)
            cv2.rectangle(
                frame, (10, y - th - 6), (18 + tw, y + baseline + 2),
                (0, 0, 0), cv2.FILLED,
            )
            color = (0, 120, 255) if "CONTRADICE" in line else (0, 220, 255)
            cv2.putText(frame, line, (14, y), font, 0.55, color, 1, cv2.LINE_AA)
            y += 22

        # "EVM" badge, top-right
        badge = "EVM"
        (tw, th), baseline = cv2.getTextSize(badge, font, 0.5, 1)
        tx, ty = frame_w - tw - 15, 25
        cv2.rectangle(frame, (tx - 5, ty - th - 5), (tx + tw + 5, ty + baseline + 3), (40, 40, 40), cv2.FILLED)
        cv2.putText(frame, badge, (tx, ty), font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    def _write_title_card(
        self,
        writer: cv2.VideoWriter,
        frame_w: int,
        frame_h: int,
        event: dict,
        clip_index: int,
        clip_total: int,
        duration_sec: float = 1.0,
        fps: float = 20.0,
    ) -> None:
        emotion = event.get("detected_emotion", "?")
        relevance = event.get("relevance_score", 0)
        timestamp = event.get("timestamp", 0.0)
        time_str = self._format_mmss(timestamp)

        card = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = f"Microexpresion {clip_index + 1}/{clip_total} - {emotion}"
        subtitle = f"Relevancia {relevance}/100  @ {time_str}"

        (tw, th), _ = cv2.getTextSize(title, font, 0.9, 2)
        cv2.putText(
            card, title, ((frame_w - tw) // 2, frame_h // 2 - 10),
            font, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
        (tw2, th2), _ = cv2.getTextSize(subtitle, font, 0.6, 1)
        cv2.putText(
            card, subtitle, ((frame_w - tw2) // 2, frame_h // 2 + 25),
            font, 0.6, (180, 180, 180), 1, cv2.LINE_AA,
        )

        n_card_frames = max(1, int(round(duration_sec * fps)))
        for _ in range(n_card_frames):
            writer.write(card)
