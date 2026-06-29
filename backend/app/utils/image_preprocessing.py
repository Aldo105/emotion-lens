"""
EmotionLens — Adaptive Image Preprocessor

Normalizes image conditions BEFORE sending to MediaPipe for detection.
This dramatically improves landmark precision in challenging conditions:
  - Low light: CLAHE + gamma correction reveal facial details
  - Dark skin tones: Adaptive CLAHE clipLimit preserves tonal range
  - Noisy webcams: Bilateral filter reduces noise while preserving edges

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
      3. Bilateral filter — Edge-preserving noise reduction
    """

    def __init__(self):
        # Pre-create CLAHE instances for each skin tone category
        # Higher clipLimit = more aggressive contrast enhancement
        self._clahe_light = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self._clahe_medium = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._clahe_dark = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

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
        # Measure brightness of the L channel after CLAHE
        brightness = float(np.mean(lab[:, :, 0])) / 255.0

        if brightness < 0.30:
            # Dark frame — apply brightening gamma
            processed = self._apply_gamma(processed, gamma=0.7)
        elif brightness > 0.80:
            # Over-exposed — apply darkening gamma
            processed = self._apply_gamma(processed, gamma=1.3)

        # ── Stage 3: Bilateral filter ────────────────────────────────
        # Reduces sensor noise while preserving sharp edges (facial
        # features, lip contours, eye corners).
        # Parameters: d=5 (small kernel), sigmaColor=50, sigmaSpace=50
        processed = cv2.bilateralFilter(processed, d=5, sigmaColor=50, sigmaSpace=50)

        return processed

    def _get_clahe(self, skin_tone: str) -> cv2.CLAHE:
        """Select CLAHE instance based on detected skin tone."""
        if skin_tone == "light":
            return self._clahe_light
        elif skin_tone == "dark":
            return self._clahe_dark
        return self._clahe_medium

    @staticmethod
    def _apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to adjust brightness.

        gamma < 1.0 → brightens (good for dark frames)
        gamma > 1.0 → darkens (good for over-exposed frames)
        """
        inv_gamma = 1.0 / gamma
        # Build a lookup table for O(1) per-pixel operation
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ], dtype=np.uint8)
        return cv2.LUT(frame, table)
