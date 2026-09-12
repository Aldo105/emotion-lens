"""
EmotionLens — Session Video Recorder

Records raw webcam frames to an MP4 file during a live analysis session.
Also stores per-frame metadata (face bounding boxes, heart rate readings)
needed by the post-session EVM renderer.

Designed for minimal overhead: cv2.VideoWriter.write() is ~1-2ms per frame,
and metadata is stored in memory until session end.
"""

import json
import os

import cv2
import numpy as np


class SessionVideoRecorder:
    """
    Records raw frames to an MP4 file for post-session EVM processing.

    Usage:
        recorder = SessionVideoRecorder(session_id, output_dir, fps=20.0)
        for frame in session_frames:
            recorder.write_frame(frame, face_bbox=detection["bbox"])
            recorder.record_hr(timestamp, bpm=72.0, confidence=0.85)
        raw_path, meta_path = recorder.close()
    """

    def __init__(
        self,
        session_id: str,
        output_dir: str,
        fps: float = 20.0,
        frame_size: tuple[int, int] = (640, 480),
    ):
        """
        Args:
            session_id: Unique session identifier.
            output_dir: Directory to write output files (e.g., data/recordings/).
            fps: Recording frame rate (must match capture FPS).
            frame_size: (width, height) of frames.
        """
        self.session_id = session_id
        self.fps = fps
        self.frame_size = frame_size
        self._frame_count = 0

        os.makedirs(output_dir, exist_ok=True)

        self.raw_video_path = os.path.join(output_dir, f"{session_id}_raw.mp4")
        self.metadata_path = os.path.join(output_dir, f"{session_id}_meta.npz")

        # Try codecs in order of preference (H.264 > MPEG-4)
        self._writer = None
        for codec in ["avc1", "H264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                self.raw_video_path, fourcc, fps, frame_size
            )
            if writer.isOpened():
                self._writer = writer
                print(f"[OK] Video recorder opened with codec '{codec}': {self.raw_video_path}")
                break
            writer.release()

        if self._writer is None:
            print(f"[ERR] Could not open VideoWriter for {self.raw_video_path}")

        # Per-frame metadata (stored in memory, flushed on close)
        self._face_bboxes: list[list[int]] = []     # [x_min, y_min, x_max, y_max] per frame
        self._hr_readings: list[dict] = []           # {timestamp, bpm, confidence} per reading
        self._timestamps: list[float] = []           # Timestamp per frame

    @property
    def is_open(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    def write_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        face_bbox: dict | None = None,
    ) -> None:
        """
        Write a single frame to the video file.

        Args:
            frame: BGR image (numpy array, uint8).
            timestamp: Session-relative timestamp in seconds.
            face_bbox: Detection bbox dict {"x_min", "y_min", "x_max", "y_max"}
                or None if no face detected.
        """
        if not self.is_open:
            return

        self._writer.write(frame)
        self._frame_count += 1
        self._timestamps.append(timestamp)

        if face_bbox is not None:
            self._face_bboxes.append([
                face_bbox["x_min"],
                face_bbox["y_min"],
                face_bbox["x_max"],
                face_bbox["y_max"],
            ])
        else:
            # Sentinel: no face detected for this frame
            self._face_bboxes.append([-1, -1, -1, -1])

    def record_hr(self, timestamp: float, bpm: float, confidence: float) -> None:
        """Store a heart rate reading for overlay on the EVM video."""
        self._hr_readings.append({
            "timestamp": timestamp,
            "bpm": bpm,
            "confidence": confidence,
        })

    def close(self) -> tuple[str, str]:
        """
        Finalize recording: close video writer and save metadata.

        Returns:
            (raw_video_path, metadata_path) tuple.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        # Save metadata as compressed numpy archive
        np.savez_compressed(
            self.metadata_path,
            face_bboxes=np.array(self._face_bboxes, dtype=np.int32),
            timestamps=np.array(self._timestamps, dtype=np.float64),
            fps=np.array([self.fps]),
            frame_size=np.array(self.frame_size),
        )

        # Save HR readings as JSON sidecar (small, human-readable)
        if self._hr_readings:
            hr_path = self.metadata_path.replace("_meta.npz", "_hr.json")
            with open(hr_path, "w") as f:
                json.dump(self._hr_readings, f)

        print(
            f"[OK] Video recording closed: {self._frame_count} frames, "
            f"{self._frame_count / max(self.fps, 1):.1f}s"
        )

        return self.raw_video_path, self.metadata_path
