"""
EmotionLens — Smart Micro-Expression Detection Engine

Detects micro-expressions using a 3-layer validation system:
  Layer 1: TEMPORAL FILTER  — Duration must be 40-500ms with onset/apex/offset
  Layer 2: CONTEXT FILTER   — Must deviate from person's baseline by 2x+
  Layer 3: RELEVANCE SCORING — Multi-factor scoring, only surface if >= 60

Micro-expressions are involuntary, brief facial expressions that reveal
concealed emotions. They last 40-500ms and typically involve specific
Action Unit combinations.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from backend.app.config import settings


@dataclass
class AUReading:
    """A single Action Unit reading at a point in time."""
    timestamp: float
    values: dict[str, float]


@dataclass
class MicroExpressionEvent:
    """A detected micro-expression event."""
    timestamp: float
    duration_ms: float
    detected_emotion: str
    dominant_emotion_at_time: str
    action_units_involved: list[str]
    relevance_score: int  # 0-100
    is_contradictory: bool
    description: str
    temporal_valid: bool = True
    context_valid: bool = True


# ── AU-to-Emotion Mapping ────────────────────────────────────────────
# Which AU combinations suggest which emotions in micro-expression context

MICRO_EXPR_PATTERNS = {
    "fear": {
        "required_aus": ["AU1", "AU4"],
        "supporting_aus": ["AU2", "AU20", "AU25"],
        "min_required": 2,
    },
    "anger": {
        "required_aus": ["AU4", "AU7"],
        "supporting_aus": ["AU23", "AU24"],
        "min_required": 2,
    },
    "disgust": {
        "required_aus": ["AU9"],
        "supporting_aus": ["AU15", "AU25"],
        "min_required": 1,
    },
    "sadness": {
        "required_aus": ["AU1", "AU15"],
        "supporting_aus": ["AU17"],
        "min_required": 2,
    },
    "surprise": {
        "required_aus": ["AU2", "AU25"],
        "supporting_aus": ["AU26"],
        "min_required": 2,
    },
    "contempt": {
        "required_aus": ["AU12", "AU14"],
        "supporting_aus": [],
        "min_required": 1,  # Unilateral AU12 is key
    },
    "stress": {
        "required_aus": ["AU23", "AU24"],
        "supporting_aus": ["AU4", "AU7"],
        "min_required": 2,
    },
}

# AUs that are commonly habitual (we track and suppress these)
HABITUAL_AUS = ["AU2", "AU4", "AU12", "AU14", "AU45"]


class MicroExpressionEngine:
    """
    Detects micro-expressions using the 3-layer smart filtering system.
    
    Maintains a rolling buffer of AU readings and analyzes temporal patterns
    to detect rapid onset-apex-offset sequences characteristic of micro-expressions.
    """

    def __init__(self):
        # Rolling buffer of recent AU readings (last ~3 seconds)
        self.buffer_duration = 3.0  # seconds
        self.au_buffer: deque[AUReading] = deque()

        # Baseline data (set by ActionUnitAnalyzer)
        self.baseline_aus: dict[str, float] = {}
        self.baseline_variability: dict[str, float] = {}  # Std deviation during baseline
        self.baseline_set = False

        # Habitual movement tracking
        self.habitual_counts: dict[str, int] = {au: 0 for au in HABITUAL_AUS}
        self.habitual_threshold = 10  # If AU fires > this many times in calibration, it's habitual
        self._calibration_frames: list[dict] = []  # Accumulated calibration AU frames

        # Active onset tracking (potential micro-expressions in progress)
        self._active_onsets: dict[str, float] = {}  # AU -> onset timestamp
        self._active_peaks: dict[str, float] = {}   # AU -> peak value

        # Recent detections (prevent duplicates)
        self._recent_detections: deque[float] = deque(maxlen=50)  # timestamps
        self._min_detection_gap = 0.3  # Min seconds between detections

        # Configuration
        self.min_duration_ms = settings.micro_expr_min_duration_ms
        self.max_duration_ms = settings.micro_expr_max_duration_ms
        self.relevance_threshold = settings.micro_expr_relevance_threshold
        self.deviation_multiplier = settings.baseline_deviation_multiplier

        # EMA smoothing state
        self._prev_smoothed: dict[str, float] | None = None

        # Latest raw AU values (for contempt asymmetry check etc.)
        self._latest_aus: dict[str, float] = {}

    def record_calibration_frame(self, aus: dict) -> None:
        """
        Record a single frame of AU values during calibration.

        For each AU in HABITUAL_AUS, increment habitual_counts if
        the AU value exceeds 0.2. Call this once per frame during
        the calibration period.

        Args:
            aus: Current AU activation values for this calibration frame.
        """
        self._calibration_frames.append(aus)
        for au in HABITUAL_AUS:
            if aus.get(au, 0.0) > 0.2:
                self.habitual_counts[au] += 1

    def set_baseline(self, baseline_aus: dict[str, float], variability: dict[str, float] = None):
        """
        Set the baseline AU values from the calibration period.

        Args:
            baseline_aus: Average AU values during calibration.
            variability: Standard deviation of AU values during calibration.
        """
        self.baseline_aus = baseline_aus
        self.baseline_variability = variability or {k: 0.05 for k in baseline_aus}
        self.baseline_set = True

        # Finalize habitual counts from accumulated calibration data
        # (record_calibration_frame already incremented counts, but
        #  re-process in case set_baseline is called with new data)
        if self._calibration_frames:
            # Reset and recount from stored frames
            self.habitual_counts = {au: 0 for au in HABITUAL_AUS}
            for frame in self._calibration_frames:
                for au in HABITUAL_AUS:
                    if frame.get(au, 0.0) > 0.2:
                        self.habitual_counts[au] += 1

    def _smooth_aus(self, current_aus: dict) -> dict:
        """
        Apply Exponential Moving Average smoothing to reduce frame-to-frame jitter.

        Args:
            current_aus: Raw AU values for the current frame.

        Returns:
            Smoothed AU values.
        """
        alpha = 0.4
        if self._prev_smoothed is None:
            self._prev_smoothed = current_aus.copy()
            return current_aus.copy()
        smoothed = {}
        for k, v in current_aus.items():
            prev = self._prev_smoothed.get(k, v)
            smoothed[k] = alpha * v + (1 - alpha) * prev
        self._prev_smoothed = smoothed.copy()
        return smoothed

    def analyze(
        self,
        current_aus: dict[str, float],
        timestamp: float,
        dominant_emotion: str = "neutral",
        camera_quality_score: float = 1.0,
    ) -> MicroExpressionEvent | None:
        """
        Analyze current AU readings for micro-expression patterns.

        Args:
            current_aus: Current AU activation values.
            timestamp: Current timestamp in seconds.
            dominant_emotion: The currently displayed dominant emotion.
            camera_quality_score: Quality factor (0-1) from the camera; penalizes
                detections when image quality is poor.

        Returns:
            MicroExpressionEvent if a micro-expression was detected, else None.
        """
        # Store latest raw AU values (used by contempt asymmetry check)
        self._latest_aus = current_aus.copy()

        # Apply EMA smoothing before buffering
        smoothed_aus = self._smooth_aus(current_aus)

        # Add to rolling buffer
        reading = AUReading(timestamp=timestamp, values=smoothed_aus)
        self.au_buffer.append(reading)

        # Trim buffer to window duration
        cutoff = timestamp - self.buffer_duration
        while self.au_buffer and self.au_buffer[0].timestamp < cutoff:
            self.au_buffer.popleft()

        # Need at least a few readings to detect patterns
        if len(self.au_buffer) < 5:
            return None

        # Check for completed micro-expressions (offset detected)
        event = self._detect_micro_expression(timestamp, dominant_emotion, camera_quality_score)
        return event

    def _detect_micro_expression(
        self,
        timestamp: float,
        dominant_emotion: str,
        camera_quality_score: float = 1.0,
    ) -> MicroExpressionEvent | None:
        """
        Core detection logic: look for rapid onset-apex-offset patterns.
        """
        # Analyze each trackable AU for onset/offset patterns
        aus_to_track = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU9",
                        "AU12", "AU14", "AU15", "AU17", "AU20",
                        "AU23", "AU24", "AU25", "AU26"]

        # Collect all (au_name, duration, peak_value) from all peaks across all AUs
        all_peak_candidates: list[tuple[str, float, float]] = []

        for au_name in aus_to_track:
            peaks = self._check_au_pattern(au_name, timestamp)
            if peaks:
                for duration, peak_value in peaks:
                    all_peak_candidates.append((au_name, duration, peak_value))

        if not all_peak_candidates:
            return None

        # Group peaks by proximity and build candidate events
        # For simplicity, collect activated AUs and their best durations
        activated_aus = []
        activation_durations = []
        seen_aus: dict[str, tuple[float, float]] = {}  # au -> (best_duration, best_peak)

        for au_name, duration, peak_value in all_peak_candidates:
            if au_name not in seen_aus or peak_value > seen_aus[au_name][1]:
                seen_aus[au_name] = (duration, peak_value)

        for au_name, (duration, _peak) in seen_aus.items():
            activated_aus.append(au_name)
            activation_durations.append(duration)

        if not activated_aus:
            return None

        # ── Layer 1: Temporal Filter ─────────────────────────────────
        avg_duration_ms = np.mean(activation_durations) * 1000
        if avg_duration_ms < self.min_duration_ms or avg_duration_ms > self.max_duration_ms:
            return None

        # ── Gap check (moved before relevance computation) ───────────
        if self._recent_detections and (timestamp - self._recent_detections[-1]) < self._min_detection_gap:
            return None

        # ── Layer 2: Context Filter ──────────────────────────────────
        if self.baseline_set:
            significant_aus = []
            for au_name in activated_aus:
                if self._exceeds_baseline(au_name):
                    if not self._is_habitual(au_name):
                        significant_aus.append(au_name)
            
            if not significant_aus:
                return None
            activated_aus = significant_aus

        # ── Match to emotion pattern ─────────────────────────────────
        detected_emotion, pattern_strength = self._match_emotion_pattern(activated_aus)
        if detected_emotion is None:
            return None

        # ── Layer 3: Relevance Scoring ───────────────────────────────
        relevance = self._compute_relevance_score(
            activated_aus=activated_aus,
            detected_emotion=detected_emotion,
            dominant_emotion=dominant_emotion,
            pattern_strength=pattern_strength,
            duration_ms=avg_duration_ms,
            camera_quality_score=camera_quality_score,
        )

        # Build the event
        is_contradictory = detected_emotion != dominant_emotion and dominant_emotion != "neutral"

        description = self._build_description(
            detected_emotion, dominant_emotion, activated_aus,
            avg_duration_ms, is_contradictory
        )

        event = MicroExpressionEvent(
            timestamp=timestamp,
            duration_ms=round(avg_duration_ms, 1),
            detected_emotion=detected_emotion,
            dominant_emotion_at_time=dominant_emotion,
            action_units_involved=activated_aus,
            relevance_score=relevance,
            is_contradictory=is_contradictory,
            description=description,
            temporal_valid=True,
            context_valid=True,
        )

        # Only surface if above threshold (but log all)
        if relevance >= self.relevance_threshold:
            self._recent_detections.append(timestamp)
            return event

        return None  # Below threshold, not surfaced

    def _check_au_pattern(self, au_name: str, current_time: float) -> list[tuple[float, float]] | None:
        """
        Check if an AU shows onset-apex-offset patterns in the buffer using
        local peak detection via scipy find_peaks.
        
        Returns a list of (duration, peak_value) tuples for each valid peak,
        or None if no valid peaks found.
        """
        if len(self.au_buffer) < 5:
            return None

        # Get AU values over time from buffer
        times = []
        values = []
        for reading in self.au_buffer:
            times.append(reading.timestamp)
            values.append(reading.values.get(au_name, 0.0))

        values = np.array(values)
        times = np.array(times)

        if len(values) < 5:
            return None

        # ── Dynamic per-AU threshold ─────────────────────────────────
        if self.baseline_set:
            baseline_val = self.baseline_aus.get(au_name, 0.0)
            baseline_var = self.baseline_variability.get(au_name, 0.05)
            threshold = baseline_val + baseline_var * 2.5
            threshold = max(threshold, 0.15)  # Absolute minimum
        else:
            threshold = 0.3  # Fallback before calibration

        # ── Find ALL local peaks above threshold ─────────────────────
        peak_indices, properties = find_peaks(values, height=threshold, distance=2)

        if len(peak_indices) == 0:
            return None

        results: list[tuple[float, float]] = []

        for peak_idx in peak_indices:
            peak_val = values[peak_idx]

            # Check onset (values before peak are low)
            if peak_idx < 2:
                continue
            pre_peak = values[:peak_idx]
            pre_mean = np.mean(pre_peak[-3:]) if len(pre_peak) >= 3 else np.mean(pre_peak)

            # Check offset (values after peak are low)  
            post_peak = values[peak_idx + 1:]
            if len(post_peak) < 2:
                # Peak is at the end — offset hasn't happened yet
                continue
            post_mean = np.mean(post_peak[:3]) if len(post_peak) >= 3 else np.mean(post_peak)

            # Verify onset-apex-offset pattern
            onset_ratio = peak_val / (pre_mean + 1e-6)
            offset_ratio = peak_val / (post_mean + 1e-6)

            if onset_ratio > 2.0 and offset_ratio > 2.0:
                # Pattern detected — compute duration
                # Find onset point (first crossing above threshold/2)
                onset_time = times[peak_idx]
                for i in range(peak_idx - 1, -1, -1):
                    if values[i] < threshold / 2:
                        onset_time = times[i]
                        break

                # Find offset point
                offset_time = times[peak_idx]
                for i in range(peak_idx + 1, len(values)):
                    if values[i] < threshold / 2:
                        offset_time = times[i]
                        break

                duration = offset_time - onset_time
                if duration > 0:
                    results.append((duration, float(peak_val)))

        return results if results else None

    def _exceeds_baseline(self, au_name: str) -> bool:
        """Check if current AU activation exceeds baseline by the deviation multiplier."""
        if not self.baseline_set:
            return True

        # Get recent peak value for this AU
        recent_values = [r.values.get(au_name, 0.0) for r in list(self.au_buffer)[-5:]]
        current_peak = max(recent_values) if recent_values else 0.0

        baseline_val = self.baseline_aus.get(au_name, 0.0)
        baseline_var = self.baseline_variability.get(au_name, 0.05)

        threshold = baseline_val + (baseline_var * self.deviation_multiplier)
        return current_peak > threshold

    def _is_habitual(self, au_name: str) -> bool:
        """Check if an AU activation is a habitual movement for this person."""
        if au_name in self.habitual_counts:
            return self.habitual_counts[au_name] >= self.habitual_threshold
        return False

    def _match_emotion_pattern(self, activated_aus: list[str]) -> tuple[str | None, float]:
        """
        Match activated AUs to known micro-expression emotion patterns.
        Returns (emotion_name, pattern_strength) or (None, 0.0).
        """
        best_emotion = None
        best_strength = 0.0

        for emotion, pattern in MICRO_EXPR_PATTERNS.items():
            required = pattern["required_aus"]
            supporting = pattern["supporting_aus"]
            min_req = pattern["min_required"]

            # Count matching required AUs
            req_matches = sum(1 for au in required if au in activated_aus)
            sup_matches = sum(1 for au in supporting if au in activated_aus)

            if req_matches >= min_req:
                # ── Contempt asymmetry check ─────────────────────────
                if emotion == "contempt":
                    face_symmetry = self._latest_aus.get("face_symmetry", None)
                    if face_symmetry is not None and face_symmetry >= 0.85:
                        # Face is too symmetric for contempt — skip
                        continue

                # Calculate pattern strength
                total_possible = len(required) + len(supporting)
                total_matches = req_matches + sup_matches
                strength = total_matches / max(total_possible, 1)

                if strength > best_strength:
                    best_strength = strength
                    best_emotion = emotion

        return best_emotion, best_strength

    def _compute_relevance_score(
        self,
        activated_aus: list[str],
        detected_emotion: str,
        dominant_emotion: str,
        pattern_strength: float,
        duration_ms: float,
        camera_quality_score: float = 1.0,
    ) -> int:
        """
        Compute relevance score (0-100) using multiple factors.
        """
        score = 0.0

        # Factor 1: Number of AUs involved (multi-AU = more significant)
        au_count_score = min(len(activated_aus) / 4.0, 1.0) * 25
        score += au_count_score

        # Factor 2: Pattern strength (how well AUs match known patterns)
        score += pattern_strength * 25

        # Factor 3: Contradiction with dominant emotion
        if detected_emotion != dominant_emotion and dominant_emotion != "neutral":
            score += 30  # Big boost for contradictory micro-expressions

        # Factor 4: Duration (sweet spot is 100-250ms)
        if 100 <= duration_ms <= 250:
            score += 20  # Optimal micro-expression duration
        elif 40 <= duration_ms <= 500:
            score += 10  # Acceptable range

        # Factor 5: Baseline deviation (if available)
        if self.baseline_set:
            deviations = []
            for au in activated_aus:
                baseline_val = self.baseline_aus.get(au, 0.0)
                recent = [r.values.get(au, 0.0) for r in list(self.au_buffer)[-5:]]
                peak = max(recent) if recent else 0.0
                if baseline_val > 0:
                    deviations.append(peak / baseline_val)
            if deviations:
                avg_deviation = np.mean(deviations)
                score += min(avg_deviation * 5, 15)

        # Apply camera quality penalty
        score *= camera_quality_score

        return int(np.clip(score, 0, 100))

    def _build_description(
        self,
        detected_emotion: str,
        dominant_emotion: str,
        activated_aus: list[str],
        duration_ms: float,
        is_contradictory: bool,
    ) -> str:
        """Build a human-readable description of the micro-expression."""
        au_str = "+".join(activated_aus)

        if is_contradictory:
            return (
                f"Brief flash of {detected_emotion} ({au_str}) "
                f"while displaying {dominant_emotion} "
                f"[{duration_ms:.0f}ms]"
            )
        else:
            return (
                f"Micro-expression of {detected_emotion} ({au_str}) "
                f"[{duration_ms:.0f}ms]"
            )
