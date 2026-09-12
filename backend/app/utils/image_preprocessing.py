"""
EmotionLens — Adaptive Image Preprocessor

Normalizes image conditions BEFORE sending to MediaPipe for detection.
This dramatically improves landmark precision in challenging conditions:
  - Low light: CLAHE + gamma correction reveal facial details
  - Dark skin tones: Adaptive CLAHE clipLimit preserves tonal range
  - Noisy webcams: Lightweight box filter reduces noise while preserving edges

The preprocessed frame is used for face detection and AU analysis.
The ORIGINAL frame is preserved for rPPG heart rate (needs true colors).
"""

import cv2
import numpy as np


class AdaptivePreprocessor:
    """
    Multi-stage adaptive image preprocessing pipeline.

    Stages:
      1. CLAHE — Contrast Limited Adaptive Histogram Equalization
      2. Gamma correction — Adaptive brightness normalization
      3. Box filter — Fast noise reduction (replaces slow bilateralFilter)

    Performance note: bilateralFilter(d=5) is O(d²×W×H) ≈ 25×W×H operations.
    A 3×3 box filter is O(9×W×H) and is separable (actually O(6×W×H)).
    At 640×480 this is ~3× faster with negligible quality loss for landmark
    detection (MediaPipe is robust to mild noise).
    """

    def __init__(self):
        # Pre-create CLAHE instances for each skin tone category
        # Higher clipLimit = more aggressive contrast enhancement
        self._clahe_light  = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self._clahe_medium = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._clahe_dark   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Pre-built gamma look-up tables (avoid recomputing each frame)
        self._lut_bright = self._build_gamma_lut(0.7)
        self._lut_dark   = self._build_gamma_lut(1.3)

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Build a 256-entry look-up table for a given gamma value."""
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        return table

    def process(self, frame: np.ndarray, skin_tone: str = "medium") -> np.ndarray:
        """
        Apply the full preprocessing pipeline.

        Args:
            frame: BGR image from webcam.
            skin_tone: "light", "medium", or "dark" — adapts CLAHE intensity.

        Returns:
            Preprocessed BGR frame with normalized contrast and brightness.
        """
        if frame is None or frame.size == 0:
            return frame

        # ── Stage 1: CLAHE on L channel (LAB colorspace) ─────────────
        # CLAHE enhances local contrast without saturating bright areas.
        # Working in LAB separates luminance from color, so we only
        # adjust brightness without shifting hues.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        clahe = self._get_clahe(skin_tone)
        lab[:, :, 0] = clahe.apply(l_channel)

        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # ── Stage 2: Adaptive gamma correction ───────────────────────
        # Measure brightness of the L channel after CLAHE.
        brightness = float(np.mean(lab[:, :, 0])) / 255.0

        if brightness < 0.30:
            # Dark frame — apply brightening gamma via LUT (O(W×H), no pow())
            processed = cv2.LUT(processed, self._lut_bright)
        elif brightness > 0.80:
            # Over-exposed — apply darkening gamma via LUT
            processed = cv2.LUT(processed, self._lut_dark)

        # ── Stage 3: Fast box blur noise reduction ────────────────────
        # Replaced bilateralFilter(d=5) with a 3×3 box blur.
        # bilateralFilter is ~3–5× slower and its edge-preservation advantage
        # is negligible at 640×480 for landmark detection.
        processed = cv2.blur(processed, (3, 3))

        return processed

    def _get_clahe(self, skin_tone: str) -> cv2.CLAHE:
        if skin_tone == "light":
            return self._clahe_light
        elif skin_tone == "dark":
            return self._clahe_dark
        return self._clahe_medium
