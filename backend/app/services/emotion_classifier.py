"""
EmotionLens — Emotion Classifier Service (v3)

Improved emotion classification with:
1. MediaPipe Blendshape-based classification (52 FACS-like coefficients)
2. Temporal smoothing (sliding window over last N frames)
3. Time-based hysteresis (requires sustained signal before switching)
4. Baseline subtraction (removes resting facial morphology bias)
5. Fallback to geometry-based heuristics when blendshapes unavailable

The blendshape approach is far more accurate than raw landmark geometry
because MediaPipe's internal ML model already does the heavy lifting.
"""

import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from backend.app.config import settings


# Emotion labels (9 categories)
EMOTION_LABELS = settings.emotion_labels

# FER2013 standard 7 labels (used by pre-trained CNN). This list maps the
# model's output indices, so its order and length must not change.
FER7_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Labels allowed to become the reported emotion. "fear" is excluded: measured
# against 220 labelled clips from 20 actors it never once won a segment, and
# the CNN gives it only ~2% of frames even inside genuine fear clips — it is
# the model's weakest class on FER2013 too (0.454). Reporting a label that
# never fires except by accident is worse than not offering it.
DOMINANT_LABELS = [lbl for lbl in FER7_LABELS if lbl != "fear"]

# Brief expressions the smoothed average cannot surface. Surprise peaks at
# 0.94 in its own segments yet wins only 21% of their frames, so it is
# reported as a moment worth reviewing rather than as a state.
BRIEF_LABELS = {"surprise"}

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


# Plausible emotion transitions — transitions NOT in this map are considered suspect
# and require higher confidence + longer streak to switch
PLAUSIBLE_TRANSITIONS = {
    "neutral": {"happy", "sad", "angry", "surprise", "disgust", "nervousness", "confidence"},
    "happy": {"neutral", "surprise", "nervousness"},
    "sad": {"neutral", "angry"},
    "angry": {"neutral", "sad", "disgust"},
    "surprise": {"neutral", "happy", "angry"},
    "disgust": {"neutral", "angry"},
    "nervousness": {"neutral", "confidence"},
    "confidence": {"neutral", "happy", "nervousness"},
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
        # Averaging window. Measured on labelled footage: the raw per-frame
        # argmax scores 57%, smoothing alone keeps it at 57%, but adding
        # hysteresis on a 12-frame window drops it to 50%. A 20-frame window
        # gives hysteresis a steadier signal to work from and recovers the
        # full 57%, at the cost of a slightly slower reaction.
        self.smoothing_window = 20        # Average over last N frames
        # Time-based switching (Improvement 10): require 250ms, not frame count
        self.min_switch_time_s = 0.25     # Require 250ms of sustained signal
        # New emotion must beat the current one by this margin. Measured on real
        # sessions, the gap between the top two labels has a median of 0.077,
        # because probability spreads across nine competing labels rather than
        # concentrating on one. The previous 0.15 sat at twice that median, so
        # 85% of the time no switch was permitted at all and whichever emotion
        # happened to be showing stayed put indefinitely.
        self.switch_confidence_threshold = 0.06

        # Minimum raw probability for a momentary reading to be reported as a
        # peak. 0.60 catches 5 of the 8 labelled surprise segments; lowering it
        # starts surfacing ordinary fluctuation instead of real flashes.
        self.peak_threshold = 0.60

        self._prob_history: deque[dict] = deque(maxlen=self.smoothing_window)
        self._current_emotion = "neutral"
        self._streak_start_time: float = 0.0   # When the current streak started
        self._streak_emotion = "neutral"        # What emotion is on the streak
        self._bounce_buffer = deque(maxlen=5)   # Tracks last 5 dominant emotions with timestamps

        # ── Baseline Subtraction (Problem 5) ──────────────────────────
        self._baseline_blendshapes: dict[str, float] | None = None
        self._baseline_set = False
        self._calibration_buffer: list[dict[str, float]] = []

        # Try to load CNN model
        if model_path is None:
            model_path = os.path.join(settings.model_dir, settings.emotion_model_name)

        if os.path.exists(model_path):
            self._load_cnn_model(model_path)
        else:
            print(f"[INFO] No trained model found at {model_path}")
            print("[INFO] Using blendshape/heuristic mode for emotion classification")

    # ══════════════════════════════════════════════════════════════════
    # FACE CROP PREPROCESSING (for CNN mode)
    # ══════════════════════════════════════════════════════════════════

    _CNN_IMG_SIZE = 224
    _CNN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _CNN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    @staticmethod
    def preprocess_face(frame_bgr: np.ndarray, bbox: dict) -> np.ndarray | None:
        """
        Crop the detected face and prepare it for the CNN: grayscale replicated
        to 3 channels, resized to 224x224, ImageNet-normalized, NCHW.

        The grayscale step is what matches training. FER2013 is a grayscale
        dataset and train_emotion_cnn.py applies Grayscale(num_output_channels=3),
        so the model has never seen an image whose channels differ. Feeding it
        colour put it out of distribution: measured against labelled footage,
        colour scored 27% and collapsed to "sad" on 73 of 83 frames, while
        grayscale scored 51% with predictions spread across the classes. The
        model itself is fine — 67% on the FER2013 test split.
        """
        x1 = max(0, bbox.get("x_min", 0))
        y1 = max(0, bbox.get("y_min", 0))
        x2 = bbox.get("x_max", 0)
        y2 = bbox.get("y_max", 0)

        # Tighten the detector's box by 10%. FER2013 faces are cropped close,
        # so MediaPipe's looser box leaves the expression occupying fewer
        # pixels than the model was trained on. Measured across all seven
        # labels this lifts accuracy from 39% to 50%; 20% tightening scores
        # worse (37%) because it biases everything toward "sad".
        inset_x = int((x2 - x1) * 0.10)
        inset_y = int((y2 - y1) * 0.10)
        x1, y1 = x1 + inset_x, y1 + inset_y
        x2, y2 = x2 - inset_x, y2 - inset_y

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        size = EmotionClassifier._CNN_IMG_SIZE
        resized = cv2.resize(crop, (size, size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        three = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        chw = three.transpose(2, 0, 1)
        chw = (chw - EmotionClassifier._CNN_MEAN) / EmotionClassifier._CNN_STD
        return chw[np.newaxis, ...].astype(np.float32)

    def record_calibration_frame(self, blendshapes: dict[str, float] | None) -> None:
        """Accumulate one frame of the calibration period for the baseline."""
        if blendshapes:
            self._calibration_buffer.append(blendshapes)

    def finalize_baseline(self) -> bool:
        """
        Average the accumulated calibration frames into the resting-face
        baseline, mirroring what ActionUnitAnalyzer.set_baseline does.

        A single frame used to be passed in directly, which made a ~50ms
        sample taken at an arbitrary instant define the subject's resting
        face for the entire session: a blink or a swallow at that moment
        biased every later prediction. Averaging the whole period is what
        this class's baseline subtraction always assumed it was getting.

        Returns whether a baseline could be established.
        """
        if not self._calibration_buffer:
            print("[!!] Emotion baseline not set: no blendshape frames captured")
            return False

        keys = set()
        for frame in self._calibration_buffer:
            keys.update(frame.keys())

        self._baseline_blendshapes = {
            key: float(np.mean([f[key] for f in self._calibration_buffer if key in f]))
            for key in keys
        }
        self._baseline_set = True
        n_frames = len(self._calibration_buffer)
        self._calibration_buffer.clear()
        print(f"[OK] Emotion classifier baseline set from {n_frames} frames "
              f"({len(self._baseline_blendshapes)} blendshapes)")
        return True

    def reset_baseline(self) -> None:
        """Clear the baseline and buffer so recalibration starts clean."""
        self._baseline_blendshapes = None
        self._baseline_set = False
        self._calibration_buffer.clear()

    def _subtract_baseline(self, blendshapes: dict[str, float]) -> dict[str, float]:
        """
        Subtract calibration baseline from current blendshapes.
        Values are clamped to [0, 1] — we only care about activations
        above the resting level.
        """
        if not self._baseline_set or self._baseline_blendshapes is None:
            return blendshapes

        adjusted = {}
        for key, value in blendshapes.items():
            baseline_val = self._baseline_blendshapes.get(key, 0.0)
            # Only keep activation above baseline; clamp to [0, 1]
            adjusted[key] = max(0.0, min(1.0, value - baseline_val))
        return adjusted

    def _load_cnn_model(self, model_path: str):
        """Load the trained PyTorch CNN model."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models

            model = models.mobilenet_v2(weights=None)
            # 7 classes, not 9: "nervousness"/"confidence" have no equivalent
            # in any public FER dataset -- they're merged in from the AU/
            # blendshape heuristic at inference time (see _merge_cnn_with_derived).
            model.classifier[1] = nn.Linear(model.last_channel, len(FER7_LABELS))
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
            cnn_result = self._predict_cnn(face_image)
            # The CNN only knows FER2013's 7 basic emotions -- "nervousness"
            # and "confidence" have no equivalent in any public FER dataset,
            # so they're carved in from the AU/blendshape signal instead of
            # being learned. See _merge_cnn_with_derived.
            merged_probs = self._merge_cnn_with_derived(
                cnn_result["probabilities"], action_units, blendshapes
            )
            top_emotion = max(merged_probs, key=merged_probs.get)
            raw_result = {
                "emotion": top_emotion,
                "confidence": merged_probs[top_emotion],
                "probabilities": merged_probs,
                "model_confidence": cnn_result["model_confidence"],
                "mode": "cnn",
            }
        elif blendshapes is not None and len(blendshapes) > 0:
            # Subtract baseline before scoring (Problem 5 fix)
            adjusted_bs = self._subtract_baseline(blendshapes)
            raw_result = self._predict_blendshapes(adjusted_bs, blendshapes)
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
            "peak": self._detect_peak(raw_result["probabilities"], final_emotion),
        }

    def _detect_peak(self, raw_probs: dict, dominant: str) -> dict | None:
        """
        A brief, strong reading that the dominant emotion will not show.

        Surprise is the clearest case: measured on labelled footage it wins 21%
        of the frames in its own segments, peaking at 0.94, yet never becomes
        the dominant emotion because the face is neutral for the rest of the
        clip. Averaging over 20 frames is what makes the dominant reading
        stable, and it is also what buries expressions that only last a moment.
        Reporting the peak separately keeps both: a steady state and the
        flashes that cross it.

        Returns None unless the peak differs from what is already displayed —
        repeating the dominant emotion would be noise, not information.
        """
        if not raw_probs:
            return None

        candidato = max(
            (k for k in raw_probs if k in DOMINANT_LABELS),
            key=lambda k: raw_probs[k],
            default=None,
        )
        if candidato is None or candidato == dominant:
            return None
        if raw_probs[candidato] < self.peak_threshold:
            return None

        pico = {
            "emotion": candidato,
            "confidence": round(raw_probs[candidato], 3),
            "review": candidato in BRIEF_LABELS,
        }
        if pico["review"]:
            # Redactado como observación, no como diagnóstico: lo único que
            # consta es que hubo un cambio facial compatible con sorpresa, y
            # quién decide qué fue es quien revisa el vídeo.
            pico["note"] = (
                "Cambio facial breve compatible con sorpresa. No se reporta como "
                "estado porque dura menos de lo que el promedio puede sostener; "
                "conviene revisar este momento en la grabación."
            )
        return pico

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

    def _validate_transition(
        self, current_emotion: str, candidate_emotion: str, candidate_confidence: float
    ) -> bool:
        """
        Validate whether a transition from current_emotion to candidate_emotion
        is allowed, considering plausibility and anti-bounce logic.
        """
        plausible = candidate_emotion in PLAUSIBLE_TRANSITIONS.get(current_emotion, set())

        if not plausible:
            margin = candidate_confidence - self._prob_history[-1].get(current_emotion, 0.0)
            # Implausible transitions need double the confidence and double the time
            if margin <= (self.switch_confidence_threshold * 2):
                return False
            # Also require the streak to have lasted at least 2x the normal time
            now = time.time()
            if (now - self._streak_start_time) < (self.min_switch_time_s * 2):
                return False

        # ── Anti-bounce check (Bug 2 fix) ────────────────────────────
        # Only reject bounces that happened within the last 3.0 seconds.
        # Previously, the bounce buffer had no time expiry, permanently
        # blocking return to any previous emotion.
        now = time.time()
        bounce_window = 3.0  # seconds

        for prev_emotion, prev_time in reversed(self._bounce_buffer):
            if now - prev_time > bounce_window:
                break  # Older entries are expired — stop checking
            if prev_emotion == candidate_emotion:
                # This emotion was left within the last 3 seconds — likely a bounce
                return False

        return True

    def _apply_hysteresis(self, smoothed_probs: dict) -> tuple[str, float]:
        """
        Prevent rapid emotion switching using hysteresis.
        The current emotion "sticks" unless a new emotion consistently
        dominates for a sustained time period with enough margin.

        Only the seven FER labels can become the dominant emotion.
        "nervousness" and "confidence" are still reported, but they are the
        project's own heuristics over blendshapes with no published prototype
        behind them (see references.py), so letting them outrank a trained
        model's output would present the two as equally grounded. It also
        measurably hurt: on labelled footage "confidence" took one segment
        from angry and another from fear outright.
        """
        now = time.time()
        eligible = {k: v for k, v in smoothed_probs.items() if k in DOMINANT_LABELS}
        if not eligible:
            eligible = smoothed_probs
        top_emotion = max(eligible, key=eligible.get)
        top_confidence = eligible[top_emotion]
        current_confidence = smoothed_probs.get(self._current_emotion, 0.0)

        # Track streak using real time (Improvement 10)
        if top_emotion == self._streak_emotion:
            # Same emotion continues — streak_start_time stays
            pass
        else:
            self._streak_emotion = top_emotion
            self._streak_start_time = now

        streak_duration = now - self._streak_start_time

        # Switch conditions:
        # 1. New emotion must beat current by threshold margin
        # 2. Must be sustained for min_switch_time_s (time-based, not frame-based)
        margin = top_confidence - current_confidence
        should_switch = (
            streak_duration >= self.min_switch_time_s
            and margin > self.switch_confidence_threshold
        )

        # Also switch when the displayed emotion is no longer even a runner-up.
        # The previous escape hatch required it to fall below 0.05, which with
        # nine competing labels almost never happened, so a stuck emotion had no
        # way back out.
        ranked = sorted(eligible, key=eligible.get, reverse=True)
        if self._current_emotion not in ranked[:2]:
            should_switch = True

        if should_switch and top_emotion != self._current_emotion:
            if not self._validate_transition(self._current_emotion, top_emotion, top_confidence):
                should_switch = False
            else:
                self._bounce_buffer.append((self._current_emotion, now))
                self._current_emotion = top_emotion

        return self._current_emotion, smoothed_probs.get(self._current_emotion, 0.0)

    # ══════════════════════════════════════════════════════════════════
    # BLENDSHAPE-BASED CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════

    def _predict_blendshapes(
        self,
        blendshapes: dict[str, float],
        raw_blendshapes: dict[str, float] | None = None,
    ) -> dict:
        """
        Classify emotion using MediaPipe's 52 blendshape coefficients.
        Much more accurate than geometry-based heuristics because MediaPipe's
        internal ML model already trained on millions of faces.

        Args:
            blendshapes: Baseline-subtracted blendshape values.
            raw_blendshapes: Original (non-subtracted) values, used to compute
                total activation for neutral scoring.
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
        # Use baseline-subtracted values to measure "extra" activation above rest.
        # (Problem 4 fix): raised cap from 0.4 → 0.65 so fully neutral faces
        # can reach ~80–90% confidence instead of being capped at ~30%.
        all_activations = [v for k, v in blendshapes.items() if k != "_neutral"]
        total_activation = sum(all_activations) if all_activations else 0

        # Stronger neutral signal: starts at 0.65, drops with facial activation
        scores["neutral"] = max(0.0, 0.65 - total_activation * 0.12)

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

        # 7 classes -- see the note in _load_cnn_model / predict() about
        # why "nervousness"/"confidence" aren't part of the CNN's output.
        prob_dict = {label: float(probs_np[i]) for i, label in enumerate(FER7_LABELS)}
        top_idx = int(np.argmax(probs_np))
        top_emotion = FER7_LABELS[top_idx]
        top_confidence = float(probs_np[top_idx])

        entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))
        max_entropy = np.log(len(FER7_LABELS))
        model_confidence = float(1.0 - (entropy / max_entropy))

        return {
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "model_confidence": model_confidence,
            "mode": "cnn",
        }

    # ══════════════════════════════════════════════════════════════════
    # CNN + AU/BLENDSHAPE HYBRID MERGE
    # ══════════════════════════════════════════════════════════════════

    def _derived_nervousness_confidence(
        self,
        action_units: dict[str, float] | None,
        blendshapes: dict[str, float] | None,
    ) -> tuple[float, float]:
        """
        Estimate nervousness/confidence "activation" in [0,1] from AUs or
        blendshapes -- same signals _predict_heuristic / _predict_blendshapes
        already use for these two labels, since no public FER dataset has
        them and the CNN can't learn them.
        """
        if blendshapes:
            bs = self._subtract_baseline(blendshapes)

            def b(name: str) -> float:
                return bs.get(name, 0.0)

            nervousness = (
                b("mouthPressLeft") * 0.15 + b("mouthPressRight") * 0.15 +
                b("mouthDimpleLeft") * 0.10 + b("mouthDimpleRight") * 0.10 +
                b("eyeSquintLeft") * 0.10 + b("eyeSquintRight") * 0.10 +
                b("jawClench") * 0.15 + b("mouthPucker") * 0.15
            )
            tension_sum = (
                b("browDownLeft") + b("browDownRight") +
                b("mouthPressLeft") + b("mouthPressRight") + b("jawClench")
            )
            relaxed = max(0.0, 1.0 - tension_sum * 0.5)
            eye_open = (b("eyeWideLeft") + b("eyeWideRight")) * 0.5
            confidence = relaxed * 0.7 + eye_open * 0.1
        elif action_units:
            def au(name: str) -> float:
                return action_units.get(name, 0.0)

            nervousness = (
                au("AU14") * 0.25 + au("AU24") * 0.30 +
                au("AU7") * 0.20 + au("blink_rate") * 0.25
            )
            low_brow_tension = max(0.0, 1.0 - au("AU4"))
            low_lip_tension = max(0.0, 1.0 - au("AU24"))
            confidence = (
                low_brow_tension * 0.30 + low_lip_tension * 0.25 +
                au("gaze_stability") * 0.45
            )
        else:
            return 0.0, 0.0

        return float(np.clip(nervousness, 0.0, 1.0)), float(np.clip(confidence, 0.0, 1.0))

    def _merge_cnn_with_derived(
        self,
        cnn_probs: dict[str, float],
        action_units: dict[str, float] | None,
        blendshapes: dict[str, float] | None,
    ) -> dict[str, float]:
        """
        Fold the AU/blendshape-derived nervousness/confidence signal into
        the CNN's 7-class distribution to produce the full 9-label output.

        When there's no nervousness/confidence signal, the CNN's 7 classes
        pass through almost unchanged. When there is, probability mass is
        carved out of the 7 CNN classes (proportionally, so the CNN's
        relative ranking among them is preserved) and handed to whichever
        of nervousness/confidence is more strongly indicated. The reserved
        share is capped at 0.5 so a strong nervousness/confidence signal
        can't fully drown out what the CNN actually saw in the face.
        """
        nervousness, confidence = self._derived_nervousness_confidence(action_units, blendshapes)
        total_signal = nervousness + confidence

        reserved = min(0.5, total_signal * 0.5)
        scale = 1.0 - reserved

        merged = {label: prob * scale for label, prob in cnn_probs.items()}
        if total_signal > 1e-6:
            merged["nervousness"] = reserved * (nervousness / total_signal)
            merged["confidence"] = reserved * (confidence / total_signal)
        else:
            merged["nervousness"] = 0.0
            merged["confidence"] = 0.0

        total = sum(merged.values())
        if total > 0:
            merged = {label: value / total for label, value in merged.items()}
        return merged

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
