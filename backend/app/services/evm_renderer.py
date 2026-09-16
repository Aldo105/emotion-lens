"""
EmotionLens — Eulerian Video Magnification Renderer

Post-processes a raw session video to produce a downloadable MP4 where
skin color changes from blood flow (pulse) are clearly visible to the
naked eye.

Algorithm (adapted from Wu et al., 2012 — MIT CSAIL):
  1. Read raw video and extract a stable face ROI using stored bounding boxes
  2. Build a Gaussian pyramid (4 levels) for the face region each frame
  3. Collect the lowest pyramid level across ALL frames into a temporal stack
  4. Apply zero-phase Butterworth bandpass filter (0.8–2.0 Hz) on the
     chrominance channels (I, Q in YIQ color space) along the time axis
  5. Amplify the filtered signal by α (40×)
  6. Reconstruct by upsampling and adding back to the original frame
  7. Feather the edges for natural blending
  8. Write to output MP4

This runs OFFLINE after the session ends.  A 10-minute session at 20 FPS
(12,000 frames) takes ~30–60 seconds on a modern CPU.
"""

import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# COLOR SPACE CONVERSIONS (BGR ↔ YIQ)
# ══════════════════════════════════════════════════════════════════════

# YIQ separates luminance (Y) from chrominance (I, Q).
# EVM amplifies only I and Q so lighting noise (in Y) is not magnified.

_BGR2RGB = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)

_RGB2YIQ = np.array([
    [0.299,  0.587,  0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523,  0.312],
], dtype=np.float32)

_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


def _bgr_to_yiq(bgr_f32: np.ndarray) -> np.ndarray:
    """BGR float32 [0,1] → YIQ float32."""
    rgb = bgr_f32 @ _BGR2RGB.T
    return rgb @ _RGB2YIQ.T


def _yiq_to_bgr(yiq: np.ndarray) -> np.ndarray:
    """YIQ float32 → BGR float32 [0,1]."""
    rgb = yiq @ _YIQ2RGB.T
    return rgb @ _BGR2RGB.T


# ══════════════════════════════════════════════════════════════════════
# GAUSSIAN PYRAMID
# ══════════════════════════════════════════════════════════════════════

def _build_pyramid_down(img: np.ndarray, levels: int) -> np.ndarray:
    """Downsample `img` through `levels` of cv2.pyrDown; return the last."""
    current = img
    for _ in range(levels):
        current = cv2.pyrDown(current)
    return current


def _upsample(small: np.ndarray, levels: int, target_h: int, target_w: int) -> np.ndarray:
    """Upsample `small` through `levels` of cv2.pyrUp, trim to target size."""
    current = small
    for _ in range(levels):
        current = cv2.pyrUp(current)
    return current[:target_h, :target_w]


# ══════════════════════════════════════════════════════════════════════
# FEATHERED EDGE MASK
# ══════════════════════════════════════════════════════════════════════

def _create_feather_mask(h: int, w: int, border: int = 15) -> np.ndarray:
    """
    Create a float32 mask (h, w, 1) that is 1.0 in the center and fades
    to 0.0 at the edges over `border` pixels.  Used to blend the EVM
    region smoothly into the original frame.
    """
    mask = np.ones((h, w), dtype=np.float32)

    for i in range(border):
        alpha = i / border
        mask[i, :] = min(mask[i, :].min(), alpha)
        mask[h - 1 - i, :] = min(mask[h - 1 - i, :].min(), alpha)
        mask[:, i] = np.minimum(mask[:, i], alpha)
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

    return mask[:, :, np.newaxis]   # (h, w, 1) for broadcasting


# ══════════════════════════════════════════════════════════════════════
# PROGRESS TRACKER
# ══════════════════════════════════════════════════════════════════════

def _write_progress(progress_path: str, status: str, progress: float = 0.0,
                    phase: str = ""):
    """
    Write rendering progress to a JSON file for the status endpoint.

    Carries the start time forward across writes so the endpoint can derive a
    remaining-time estimate from how long the work so far actually took, which
    is the only honest basis for one: render speed depends on the machine.
    """
    now = time.time()
    started_at = now
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            started_at = json.load(f).get("started_at", now)
    except Exception:
        pass

    try:
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({
                "status": status,
                "progress": round(progress, 3),
                "phase": phase,
                "started_at": started_at,
                "updated_at": now,
            }, f)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# MEMORY BUDGET
# ══════════════════════════════════════════════════════════════════════

def _available_memory_mb() -> float | None:
    """Physical memory currently available, or None if it can't be determined."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / (1024 * 1024)
            return None

        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None
    return None


def _chunk_frames(fps: float, pyr_h: int, pyr_w: int, total_frames: int) -> int:
    """
    How many frames to hold in memory at once.

    Sized from a slice of the machine's free memory so a laptop under pressure
    works in smaller blocks instead of failing, and capped well below what is
    available because this runs while the app may still be serving a session.
    """
    avail = _available_memory_mb()
    if avail is None:
        budget_mb = 48.0
    else:
        budget_mb = max(16.0, min(128.0, avail * 0.10))

    bytes_per_frame = pyr_h * pyr_w * 3 * 4
    frames = int((budget_mb * 1024 * 1024) / max(bytes_per_frame, 1))

    # At least a few seconds, or the bandpass has too little signal to work on.
    return max(int(fps * 10), min(frames, total_frames))


# ══════════════════════════════════════════════════════════════════════
# MAIN RENDERER
# ══════════════════════════════════════════════════════════════════════

class EVMRenderer:
    """
    Offline Eulerian Video Magnification renderer.

    Processes the video in overlapping temporal blocks: each block collects the
    lowest pyramid level for its frames, bandpasses them, then re-reads and
    writes just that block's output. Memory therefore depends on the block
    size rather than the length of the session — the previous whole-video
    stacks grew with duration, so an hour-long interview needed roughly a
    gigabyte where a one-minute test needed 18 MB.

    The blocks overlap because a Butterworth filter rings at the edges of its
    input; the padding is filtered and then discarded, so seams between blocks
    are not visible.
    """

    def __init__(
        self,
        amplification: float = 40.0,
        freq_low: float = 0.8,
        freq_high: float = 2.0,
        pyramid_levels: int = 4,
        delete_raw: bool = True,
    ):
        self.amplification = amplification
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.pyramid_levels = pyramid_levels
        self.delete_raw = delete_raw

    def render(
        self,
        raw_video_path: str,
        metadata_path: str,
        output_path: str,
    ) -> str | None:
        """
        Render the EVM-magnified video.

        Args:
            raw_video_path: Path to the raw session video (MP4).
            metadata_path: Path to the .npz metadata (face bboxes, etc.).
            output_path: Path for the output EVM video.

        Returns:
            The output_path on success, or None on failure.
        """
        if not SCIPY_AVAILABLE:
            print("[ERR] scipy required for EVM rendering but not installed")
            return None

        progress_path = output_path.replace(".mp4", "_progress.json")
        _write_progress(progress_path, "rendering", 0.0, "initializing")

        try:
            result = self._render_impl(raw_video_path, metadata_path,
                                       output_path, progress_path)
            if result:
                _write_progress(progress_path, "ready", 1.0, "done")

                # Clean up raw files
                if self.delete_raw:
                    self._cleanup(raw_video_path, metadata_path)
            else:
                _write_progress(progress_path, "failed", 0.0, "error")

            return result

        except Exception as e:
            print(f"[ERR] EVM rendering failed: {e}")
            traceback.print_exc()
            _write_progress(progress_path, "failed", 0.0, str(e))
            return None

    def _render_impl(
        self,
        raw_video_path: str,
        metadata_path: str,
        output_path: str,
        progress_path: str,
    ) -> str | None:
        # ── Load metadata ────────────────────────────────────────────
        meta = np.load(metadata_path, allow_pickle=True)
        face_bboxes = meta["face_bboxes"]       # (N, 4) int32
        fps_arr = meta["fps"]
        fps = float(fps_arr[0]) if len(fps_arr) > 0 else 20.0

        # ── Open raw video ───────────────────────────────────────────
        cap = cv2.VideoCapture(raw_video_path)
        if not cap.isOpened():
            print(f"[ERR] Cannot open raw video: {raw_video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames < 60:
            print(f"[WARN] Video too short for EVM ({total_frames} frames)")
            cap.release()
            return None

        # ── Compute stable median face bbox ──────────────────────────
        valid_mask = face_bboxes[:, 0] >= 0
        if valid_mask.sum() < 30:
            print("[WARN] Not enough face detections for EVM rendering")
            cap.release()
            return None

        valid_bboxes = face_bboxes[valid_mask]
        median_bbox = np.median(valid_bboxes, axis=0).astype(int)

        # Expand bbox by 15% for context
        bw = median_bbox[2] - median_bbox[0]
        bh = median_bbox[3] - median_bbox[1]
        expand_x = int(bw * 0.15)
        expand_y = int(bh * 0.15)

        x1 = max(0, median_bbox[0] - expand_x)
        y1 = max(0, median_bbox[1] - expand_y)
        x2 = min(frame_w, median_bbox[2] + expand_x)
        y2 = min(frame_h, median_bbox[3] + expand_y)

        roi_h = y2 - y1
        roi_w = x2 - x1

        if roi_h < 30 or roi_w < 30:
            print("[WARN] Face ROI too small for EVM rendering")
            cap.release()
            return None

        print(f"[EVM] Rendering {total_frames} frames, "
              f"face ROI: {roi_w}×{roi_h}, "
              f"amplification: {self.amplification}×")

        # Determine pyramid level dimensions
        test_roi = np.zeros((roi_h, roi_w, 3), dtype=np.float32)
        test_down = _build_pyramid_down(test_roi, self.pyramid_levels)
        pyr_h, pyr_w = test_down.shape[:2]

        # ── Design the Butterworth bandpass up front ─────────────────
        nyquist = fps / 2.0
        low = self.freq_low / nyquist
        high = min(self.freq_high / nyquist, 0.95)  # Stay below Nyquist

        if low >= high or low <= 0:
            print(f"[WARN] Invalid filter band: {low:.3f}–{high:.3f} (normalized)")
            cap.release()
            return None

        b, a = butter(3, [low, high], btype="band")

        # ── Size the temporal blocks ─────────────────────────────────
        chunk = _chunk_frames(fps, pyr_h, pyr_w, total_frames)
        # Two seconds of padding on each side, filtered then discarded, so the
        # filter's edge transient never lands on a frame that gets written.
        overlap = min(int(fps * 2), max(chunk // 4, 1))
        block_mb = (chunk + 2 * overlap) * pyr_h * pyr_w * 3 * 4 / (1024 * 1024)
        print(f"[EVM] Block size: {chunk} frames (+{overlap} overlap), "
              f"~{block_mb:.1f} MB per block")

        _write_progress(progress_path, "rendering", 0.05, "processing in blocks")

        # Open output video writer
        writer = None
        for codec in ["avc1", "H264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (frame_w, frame_h)
            )
            if writer.isOpened():
                break
            writer.release()
            writer = None

        if writer is None:
            print("[ERR] Cannot open VideoWriter for EVM output")
            cap.release()
            return None

        # Create feathered edge mask for smooth blending
        feather_mask = _create_feather_mask(roi_h, roi_w, border=20)

        # Load HR data for overlay (if available)
        hr_data = self._load_hr_data(raw_video_path, metadata_path)

        # ── Process block by block ───────────────────────────────────
        for start in range(0, total_frames, chunk):
            end = min(start + chunk, total_frames)
            pad_start = max(0, start - overlap)
            pad_end = min(total_frames, end + overlap)
            n_pad = pad_end - pad_start

            # Collect this block's pyramid levels (padding included)
            stack = np.zeros((n_pad, pyr_h, pyr_w, 3), dtype=np.float32)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pad_start)
            read = 0
            for i in range(n_pad):
                ret, frame = cap.read()
                if not ret:
                    break
                roi = frame[y1:y2, x1:x2].astype(np.float32) / 255.0
                level = _build_pyramid_down(_bgr_to_yiq(roi), self.pyramid_levels)
                ph = min(level.shape[0], pyr_h)
                pw = min(level.shape[1], pyr_w)
                stack[i, :ph, :pw, :] = level[:ph, :pw, :]
                read += 1

            if read == 0:
                break

            # Bandpass I and Q in place. Y is zeroed rather than filtered:
            # amplifying luminance would brighten the face instead of revealing
            # the colour change blood flow produces.
            padlen = min(3 * max(len(a), len(b)), read - 1)
            if padlen > 0:
                for ch in (1, 2):
                    stack[:read, :, :, ch] = filtfilt(
                        b, a, stack[:read, :, :, ch], axis=0, padlen=padlen
                    )
            stack[:, :, :, 0] = 0.0
            stack *= self.amplification

            # Re-read the block's own frames and write them out
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break

                idx = i - pad_start
                if idx >= read:
                    break

                upsampled = _upsample(stack[idx], self.pyramid_levels, roi_h, roi_w)

                roi_bgr = frame[y1:y2, x1:x2].astype(np.float32) / 255.0
                roi_yiq = _bgr_to_yiq(roi_bgr)
                roi_yiq += upsampled

                roi_magnified = _yiq_to_bgr(roi_yiq)
                roi_magnified = np.clip(roi_magnified * 255.0, 0, 255).astype(np.uint8)

                frame_f = frame[y1:y2, x1:x2].astype(np.float32)
                magnified_f = roi_magnified.astype(np.float32)
                blended = magnified_f * feather_mask + frame_f * (1.0 - feather_mask)
                frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

                self._draw_hr_overlay(frame, i, fps, hr_data)
                self._draw_evm_badge(frame, frame_w)
                writer.write(frame)

            del stack

            _write_progress(
                progress_path, "rendering",
                0.05 + 0.93 * (end / total_frames),
                f"frames {end}/{total_frames}"
            )

        writer.release()
        cap.release()

        # Verify output
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] EVM video rendered: {output_path} ({output_size:.1f} MB)")

        return output_path

    # ── Helpers ───────────────────────────────────────────────────────

    def _load_hr_data(
        self, raw_video_path: str, metadata_path: str
    ) -> list[dict] | None:
        """Load HR readings from the JSON sidecar if it exists."""
        hr_path = metadata_path.replace("_meta.npz", "_hr.json")
        if os.path.exists(hr_path):
            try:
                with open(hr_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _draw_hr_overlay(
        self, frame: np.ndarray, frame_idx: int, fps: float,
        hr_data: list[dict] | None
    ) -> None:
        """Draw heart rate value on the frame (bottom-left corner)."""
        if not hr_data:
            return

        current_time = frame_idx / fps
        # Find closest HR reading
        best = None
        best_dist = float("inf")
        for entry in hr_data:
            dist = abs(entry["timestamp"] - current_time)
            if dist < best_dist:
                best_dist = dist
                best = entry

        if best is None or best["confidence"] < 0.3:
            return

        bpm = int(round(best["bpm"]))
        h = frame.shape[0]

        # Draw background pill
        text = f"HR: {bpm} BPM"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        tx, ty = 15, h - 20
        cv2.rectangle(
            frame, (tx - 6, ty - th - 8), (tx + tw + 6, ty + baseline + 4),
            (0, 0, 0), cv2.FILLED
        )
        cv2.putText(frame, text, (tx, ty), font, scale, (100, 220, 255),
                     thickness, cv2.LINE_AA)

    def _draw_evm_badge(self, frame: np.ndarray, frame_w: int) -> None:
        """Draw a small 'EVM' indicator in the top-right corner."""
        text = "EVM"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        tx = frame_w - tw - 15
        ty = 25
        cv2.rectangle(
            frame, (tx - 5, ty - th - 5), (tx + tw + 5, ty + baseline + 3),
            (40, 40, 40), cv2.FILLED
        )
        cv2.putText(frame, text, (tx, ty), font, scale, (0, 200, 255),
                     thickness, cv2.LINE_AA)

    def _cleanup(self, raw_video_path: str, metadata_path: str) -> None:
        """Delete raw video and metadata files after successful render."""
        for path in [raw_video_path, metadata_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"[OK] Cleaned up: {path}")
            except Exception as e:
                print(f"[WARN] Failed to delete {path}: {e}")

        # Also clean up HR sidecar
        hr_path = metadata_path.replace("_meta.npz", "_hr.json")
        try:
            if os.path.exists(hr_path):
                os.remove(hr_path)
        except Exception:
            pass
