"""
EmotionLens — Emotion Classifier Service (v2)

Improved emotion classification with:
1. MediaPipe Blendshape-based classification (52 FACS-like coefficients)
2. Temporal smoothing (sliding window over last N frames)
3. Emotion change threshold (requires sustained signal before switching)
4. Fallback to geometry-based heuristics when blendshapes unavailable

The blendshape approach is far more accurate than raw landmark geometry
because MediaPipe's internal ML model already does the heavy lifting.
"""

import os
from collections import deque
from pathlib import Path

import numpy as np

from backend.app.config import settings


# Emotion labels (9 categories)
EMOTION_LABELS = settings.emotion_labels

# FER2013 standard 7 labels (used by pre-trained CNN)
FER7_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ── Blendshape-to-Emotion Mapping ────────────────────────────────────
# MediaPipe blendshape names map to FACS Action Units.
# We define which blendshapes contribute to each emotion and their weights.

BLENDSHAPE_EMOTION_MAP = {
    "happy": {
        "mouthSmileLeft": 0.35,
        "mouthSmileRight": 0.35,
        "cheekSquintLeft": 0.15,
        "cheekSquintRight": 0.15,
    },
    "sad": {
        "mouthFrownLeft": 0.30,
        "mouthFrownRight": 0.30,
        "browInnerUp": 0.25,
        "mouthPucker": 0.15,
    },
    "angry": {
        "browDownLeft": 0.30,
        "browDownRight": 0.30,
        "mouthPressLeft": 0.10,
        "mouthPressRight": 0.10,
        "jawForward": 0.10,
        "noseSneerLeft": 0.05,
        "noseSneerRight": 0.05,
    },
    "surprise": {
        "browOuterUpLeft": 0.15,
        "browOuterUpRight": 0.15,
        "browInnerUp": 0.20,
        "eyeWideLeft": 0.10,
        "eyeWideRight": 0.10,
        "jawOpen": 0.20,
        "mouthFunnel": 0.10,
    },
    "disgust": {
        "noseSneerLeft": 0.35,
        "noseSneerRight": 0.35,
        "mouthUpperUpLeft": 0.15,
        "mouthUpperUpRight": 0.15,
    },
    "fear": {
        "browInnerUp": 0.20,
        "browOuterUpLeft": 0.10,
        "browOuterUpRight": 0.10,
        "eyeWideLeft": 0.20,
        "eyeWideRight": 0.20,
        "mouthStretchLeft": 0.10,
        "mouthStretchRight": 0.10,
    },
    "nervousness": {
        "mouthPressLeft": 0.15,
        "mouthPressRight": 0.15,
        "mouthDimpleLeft": 0.10,
        "mouthDimpleRight": 0.10,
        "eyeSquintLeft": 0.10,
        "eyeSquintRight": 0.10,
        "jawClench": 0.15,
        "mouthPucker": 0.15,
    },
    "confidence": {
        # Confidence is the ABSENCE of tension + steady gaze
        # We handle it specially in the scoring logic
    },
}


class EmotionClassifier:
    """
    Classifies facial emotions using blendshapes or landmark heuristics.
    Includes temporal smoothing to prevent rapid emotion oscillation.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.mode = "heuristic"  # "cnn", "blendshape", or "heuristic"

        # ── Temporal Smoothing ────────────────────────────────────────
        self.smoothing_window = 12        # Average over last N frames
        self.min_switch_frames = 5        # Require N consecutive frames to switch emotion
        self.switch_confidence_threshold = 0.15  # New emotion must beat current by this margin

        self._prob_history: deque[dict] = deque(maxlen=self.smoothing_window)
        self._current_emotion = "neutral"
        self._emotion_streak = 0          # How many frames the top emotion has been consistent
        self._streak_emotion = "neutral"  # What emotion is on the streak

        # Try to load CNN model
        if model_path is None:
            model_path = os.path.join(settings.model_dir, settings.emotion_model_name)

        if os.path.exists(model_path):
            self._load_cnn_model(model_path)
        else:
            print(f"[INFO] No trained model found at {model_path}")
            print("[INFO] Using blendshape/heuristic mode for emotion classification")

    def _load_cnn_model(self, model_path: str):
        """Load the trained PyTorch CNN model."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models

            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, len(EMOTION_LABELS))
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.model = model
            self.mode = "cnn"
            print(f"[OK] Emotion CNN loaded from {model_path} ({self.device})")
        except Exception as e:
            print(f"[!!] Failed to load CNN model: {e}")
            print("[INFO] Falling back to blendshape/heuristic mode")

    def predict(
        self,
        face_image: np.ndarray = None,
        action_units: dict[str, float] = None,
        blendshapes: dict[str, float] = None,
        quality_penalty: float = 1.0,
    ) -> dict:
        """
        Predict emotion with temporal smoothing.

        Args:
            face_image: Preprocessed face crop (for CNN mode).
            action_units: Dict of AU name -> activation level.
            blendshapes: Dict of MediaPipe blendshape name -> score (0-1).
            quality_penalty: 0.0–1.0 multiplier from camera quality gate.
                Degraded frames get 0.75, reducing reported confidence.

        Returns:
            {
                "emotion": str,
                "confidence": float,
                "probabilities": dict,
                "model_confidence": float,
                "mode": str,
            }
        """
        # Step 1: Get raw probabilities from the best available source
        if self.mode == "cnn" and face_image is not None:
            raw_result = self._predict_cnn(face_image)
        elif blendshapes is not None and len(blendshapes) > 0:
            raw_result = self._predict_blendshapes(blendshapes)
        elif action_units is not None:
            raw_result = self._predict_heuristic(action_units)
        else:
            return self._empty_prediction()

        # Step 2: Apply temporal smoothing
        smoothed = self._apply_temporal_smoothing(raw_result["probabilities"])

        # Step 3: Apply emotion switching logic (hysteresis)
        final_emotion, final_confidence = self._apply_hysteresis(smoothed)

        # Step 4: Apply quality penalty (degraded frames get lower confidence)
        final_confidence *= quality_penalty

        return {
            "emotion": final_emotion,
            "confidence": final_confidence,
            "probabilities": smoothed,
            "model_confidence": raw_result["model_confidence"] * quality_penalty,
            "mode": raw_result["mode"],
        }

    # ══════════════════════════════════════════════════════════════════
    # TEMPORAL SMOOTHING
    # ══════════════════════════════════════════════════════════════════

    def _apply_temporal_smoothing(self, current_probs: dict) -> dict:
        """
        Average probabilities over the last N frames for stability.
        Uses exponential weighting so recent frames matter more.
        """
        self._prob_history.append(current_probs.copy())

        if len(self._prob_history) < 2:
            return current_probs

        # Exponential weights: most recent frame has highest weight
        n = len(self._prob_history)
        weights = np.exp(np.linspace(-1.0, 0.0, n))
        weights /= weights.sum()

        smoothed = {}
        for label in EMOTION_LABELS:
            values = [h.get(label, 0.0) for h in self._prob_history]
            smoothed[label] = float(np.average(values, weights=weights))

        # Re-normalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed

    def _apply_hysteresis(self, smoothed_probs: dict) -> tuple[str, float]:
        """
        Prevent rapid emotion switching using hysteresis.
        The current emotion "sticks" unless a new emotion consistently
        dominates for several consecutive frames with enough margin.
        """
        top_emotion = max(smoothed_probs, key=smoothed_probs.get)
        top_confidence = smoothed_probs[top_emotion]
        current_confidence = smoothed_probs.get(self._current_emotion, 0.0)

        # Track streak
        if top_emotion == self._streak_emotion:
            self._emotion_streak += 1
        else:
            self._streak_emotion = top_emotion
            self._emotion_streak = 1

        # Switch conditions:
        # 1. New emotion must beat current by threshold margin
        # 2. Must be sustained for min_switch_frames
        margin = top_confidence - current_confidence
        should_switch = (
            self._emotion_streak >= self.min_switch_frames
            and margin > self.switch_confidence_threshold
        )

        # Also switch if current emotion has dropped very low
        if current_confidence < 0.05:
            should_switch = True

        if should_switch and top_emotion != self._current_emotion:
            self._current_emotion = top_emotion

        return self._current_emotion, smoothed_probs.get(self._current_emotion, 0.0)

    # ══════════════════════════════════════════════════════════════════
    # BLENDSHAPE-BASED CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════

    def _predict_blendshapes(self, blendshapes: dict[str, float]) -> dict:
        """
        Classify emotion using MediaPipe's 52 blendshape coefficients.
        Much more accurate than geometry-based heuristics because MediaPipe's
        internal ML model already trained on millions of faces.
        """
        scores = {}

        for emotion, bs_weights in BLENDSHAPE_EMOTION_MAP.items():
            if emotion == "confidence":
                continue  # Handled separately
            score = 0.0
            for bs_name, weight in bs_weights.items():
                score += blendshapes.get(bs_name, 0.0) * weight
            scores[emotion] = score

        # ── Neutral: inverse of total activation ──
        all_activations = [v for k, v in blendshapes.items() if k != "_neutral"]
        total_activation = sum(all_activations) if all_activations else 0
        # More activation = less neutral
        scores["neutral"] = max(0.0, 0.4 - total_activation * 0.08)

        # ── Confidence: relaxed face + no tension indicators ──
        tension_sum = (
            blendshapes.get("browDownLeft", 0) + blendshapes.get("browDownRight", 0) +
            blendshapes.get("mouthPressLeft", 0) + blendshapes.get("mouthPressRight", 0) +
            blendshapes.get("jawClench", 0)
        )
        relaxed = max(0.0, 1.0 - tension_sum * 0.5)
        # Eye openness indicates alertness/confidence
        eye_open = (blendshapes.get("eyeWideLeft", 0) + blendshapes.get("eyeWideRight", 0)) * 0.5
        scores["confidence"] = (relaxed * 0.7 + eye_open * 0.1) * 0.3

        # Softmax normalization with temperature
        score_values = np.array(list(scores.values()), dtype=np.float32)
        temperature = 0.3  # Lower = more decisive
        exp_scores = np.exp((score_values - np.max(score_values)) / temperature)
        probabilities = exp_scores / (np.sum(exp_scores) + 1e-10)

        prob_dict = {label: float(probabilities[i]) for i, label in enumerate(scores.keys())}

        top_emotion = max(prob_dict, key=prob_dict.get)
        top_confidence = prob_dict[top_emotion]

        sorted_probs = sorted(probabilities, reverse=True)
        model_confidence = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.5

        return {
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "model_confidence": min(1.0, model_confidence),
            "mode": "blendshape",
        }

    # ══════════════════════════════════════════════════════════════════
    # HEURISTIC (GEOMETRY) FALLBACK
    # ══════════════════════════════════════════════════════════════════

    def _predict_heuristic(self, action_units: dict[str, float]) -> dict:
        """
        Estimate emotions from geometry-based Action Unit activations.
        Only used when blendshapes are not available.
        """
        scores = {}

        def au(name: str) -> float:
            return action_units.get(name, 0.0)

        scores["happy"] = (au("AU6") * 0.4 + au("AU12") * 0.6)
        scores["sad"] = (au("AU1") * 0.35 + au("AU15") * 0.40 + au("AU17") * 0.25)
        scores["angry"] = (au("AU4") * 0.35 + au("AU7") * 0.30 + au("AU23") * 0.35)
        scores["surprise"] = (au("AU2") * 0.30 + au("AU25") * 0.30 + au("AU26") * 0.40)
        scores["disgust"] = (au("AU9") * 0.60 + au("AU15") * 0.40)
        scores["fear"] = (au("AU1") * 0.25 + au("AU2") * 0.25 +
                          au("AU4") * 0.20 + au("AU20") * 0.30)

        total_activation = sum(action_units.values()) if action_units else 0
        scores["neutral"] = max(0.0, 1.0 - (total_activation / max(len(action_units), 1)))

        scores["nervousness"] = (au("AU14") * 0.25 + au("AU24") * 0.30 +
                                  au("AU7") * 0.20 + au("blink_rate") * 0.25)

        low_brow_tension = max(0.0, 1.0 - au("AU4"))
        low_lip_tension = max(0.0, 1.0 - au("AU24"))
        gaze_stability = au("gaze_stability")
        scores["confidence"] = (low_brow_tension * 0.30 + low_lip_tension * 0.25 +
                                 gaze_stability * 0.45)

        score_values = np.array(list(scores.values()), dtype=np.float32)
        temperature = 0.5
        exp_scores = np.exp((score_values - np.max(score_values)) / temperature)
        probabilities = exp_scores / (np.sum(exp_scores) + 1e-10)

        prob_dict = {label: float(probabilities[i]) for i, label in enumerate(scores.keys())}
        top_emotion = max(prob_dict, key=prob_dict.get)
        top_confidence = prob_dict[top_emotion]

        sorted_probs = sorted(probabilities, reverse=True)
        model_confidence = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.5

        return {
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "model_confidence": min(1.0, model_confidence),
            "mode": "heuristic",
        }

    # ══════════════════════════════════════════════════════════════════
    # CNN
    # ══════════════════════════════════════════════════════════════════

    def _predict_cnn(self, face_image: np.ndarray) -> dict:
        """Run CNN inference on a preprocessed face image."""
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(face_image).float().to(self.device)
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            probs_np = probabilities.cpu().numpy()

        prob_dict = {label: float(probs_np[i]) for i, label in enumerate(EMOTION_LABELS)}
        top_idx = int(np.argmax(probs_np))
        top_emotion = EMOTION_LABELS[top_idx]
        top_confidence = float(probs_np[top_idx])

        entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))
        max_entropy = np.log(len(EMOTION_LABELS))
        model_confidence = float(1.0 - (entropy / max_entropy))

        return {
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "model_confidence": model_confidence,
            "mode": "cnn",
        }

    def _empty_prediction(self) -> dict:
        """Return an empty prediction when no input is available."""
        prob_dict = {label: 0.0 for label in EMOTION_LABELS}
        prob_dict["neutral"] = 1.0
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": prob_dict,
            "model_confidence": 0.0,
            "mode": "none",
        }
