"""
EmotionLens — Smart Micro-Expression Detection Engine

Detects micro-expressions using a 3-layer validation system:
  Layer 1: TEMPORAL FILTER  — Duration must be 40-500ms with onset/apex/offset
  Layer 2: CONTEXT FILTER   — Must deviate from person's baseline by 2x+
  Layer 3: RELEVANCE SCORING — Multi-factor scoring, only surface if >= 60

Micro-expressions are involuntary, brief facial expressions that reveal
concealed emotions. They last 40-500ms and typically involve specific
Action Unit combinations.

Onset/apex/offset is measured on each AU's deviation from the person's resting
baseline, as the full width at half maximum of the peak: the expression must
rise from rest, peak above the per-AU threshold and fall back to rest. The AUs
of one event must peak together (within COOCCURRENCE_WINDOW_S). The input AUs
are already EMA-smoothed by ActionUnitAnalyzer; this engine does not smooth
them again — see analyze().
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.services.noise_filter import NoiseState

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


@dataclass
class AUPeak:
    """One completed onset→apex→offset of a single AU."""
    au: str
    peak_time: float
    peak_value: float
    amplitude: float      # peak minus the person's resting value
    duration: float       # seconds, full width at half maximum
    offset_time: float


# The micro patterns and the emotion classifier name emotions differently
# ("anger" vs "angry", "sadness" vs "sad"). Comparing the raw strings marked a
# micro-expression as contradicting the displayed emotion even when both were
# the same emotion, which added +30 relevance and fed the trust score, the red
# flags and the report's "masking" text with false contradictions.
MICRO_TO_CLASSIFIER_LABEL: dict[str, str] = {
    "anger": "angry",
    "sadness": "sad",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    # "contempt" and "stress" have no classifier label: any non-neutral
    # displayed emotion differs from them.
}

# Displayed labels that carry no emotion to contradict.
_NON_COMPARABLE_DISPLAYED = {"neutral", "unmeasurable", "", None}


def micro_contradicts_displayed(detected_emotion: str, dominant_emotion: str | None) -> bool:
    """Whether a micro-expression contradicts the emotion shown on the face."""
    if dominant_emotion in _NON_COMPARABLE_DISPLAYED:
        return False
    equivalent = MICRO_TO_CLASSIFIER_LABEL.get(detected_emotion, detected_emotion)
    return equivalent != dominant_emotion


# ── AU-to-Emotion Mapping ────────────────────────────────────────────
# Which AU combinations suggest which emotions in micro-expression context

MICRO_EXPR_PATTERNS = {
    # AU5 (Upper Lid Raiser) is now computed (action_units.py) and added
    # below to fear/anger/surprise, which EMFACS lists it in but the code
    # previously omitted entirely — biasing those three toward false
    # negatives. See backend/app/references.py, MICRO_EXPR_PATTERNS.
    "fear": {
        # EMFACS: 1+2+4+5+7+20+25/26
        "required_aus": ["AU1", "AU4"],
        "supporting_aus": ["AU2", "AU5", "AU20", "AU25"],
        "min_required": 2,
    },
    "anger": {
        # EMFACS: 4+5+7+23
        "required_aus": ["AU4", "AU7"],
        "supporting_aus": ["AU5", "AU23", "AU24"],
        "min_required": 2,
    },
    "disgust": {
        # EMFACS: 9+15+16 (+6,11,17). AU25 was listed as supporting but
        # isn't part of the EMFACS disgust prototype — dropped.
        "required_aus": ["AU9"],
        "supporting_aus": ["AU15"],
        "min_required": 1,
    },
    "sadness": {
        # EMFACS: 1+4+15
        "required_aus": ["AU1", "AU15"],
        "supporting_aus": ["AU4", "AU17"],
        "min_required": 2,
    },
    "surprise": {
        # EMFACS: 1+2+5+26
        "required_aus": ["AU2", "AU25"],
        "supporting_aus": ["AU1", "AU5", "AU26"],
        "min_required": 2,
    },
    "contempt": {
        "required_aus": ["AU12", "AU14"],
        "supporting_aus": [],
        "min_required": 1,  # Unilateral AU12 is key
    },
    # NOT an EMFACS prototype — this AU combination is a project heuristic,
    # not a published expression pattern. Kept because downstream code
    # matches on the "stress" label, but it should be read as descriptive
    # (lip tightening + press), not as a validated stress signature.
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
        # (Problem 7 fix): percentage of calibration frames, not absolute count.
        # Previously, 10 absolute frames (0.5s at 20fps) in a 30s calibration
        # permanently suppressed the AU. Now requires >= 15% of calibration time.
        self.habitual_ratio_threshold = 0.15  # AU active in >= 15% of calibration = habitual
        self._calibration_frame_count = 0     # Total calibration frames (for percentage calc)
        self._calibration_frames: list[dict] = []  # Accumulated calibration AU frames

        # Last peak time already reported, per AU, so a peak that stays in the
        # rolling buffer is not reported again on the following frames.
        self._consumed_peaks: dict[str, float] = {}

        # A peak only counts once it has returned to rest, and only while
        # that return is recent. Keeps the event close to "now" and lets an AU
        # whose offset lands a frame or two later still join the same event.
        self.completion_window_s = 0.4
        # AUs of one expression peak together; peaks further apart than this
        # are separate movements and are not combined into one pattern.
        self.cooccurrence_window_s = 0.2
        # A buffer gap larger than this (frames skipped for yawns, scratching,
        # forced blinks or a lost face) breaks the series: values on either
        # side of the gap are not consecutive and must not form a peak.
        self.max_frame_gap_s = 0.25
        # Frames of a peak that must sit above the per-AU threshold.
        self.min_frames_above_threshold = 2

        # Recent detections (prevent duplicates)
        self._recent_detections: deque[float] = deque(maxlen=50)  # timestamps
        self._min_detection_gap = 0.3  # Min seconds between detections

        # Configuration
        self.min_duration_ms = settings.micro_expr_min_duration_ms
        self.max_duration_ms = settings.micro_expr_max_duration_ms
        self.relevance_threshold = settings.micro_expr_relevance_threshold
        self.deviation_multiplier = settings.baseline_deviation_multiplier

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

    def reset_calibration(self) -> None:
        """
        Forget the previous calibration before a recalibration.

        The calibration frames used to survive a recalibration, so the
        habitual-movement counts mixed the old and the new calibration.
        """
        self.baseline_aus = {}
        self.baseline_variability = {}
        self.baseline_set = False
        self._calibration_frames = []
        self._calibration_frame_count = 0
        self.habitual_counts = {au: 0 for au in HABITUAL_AUS}
        self._consumed_peaks = {}
        self.au_buffer.clear()

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
            # Store total frame count for percentage-based habitual check
            self._calibration_frame_count = len(self._calibration_frames)
            # Reset and recount from stored frames
            self.habitual_counts = {au: 0 for au in HABITUAL_AUS}
            for frame in self._calibration_frames:
                for au in HABITUAL_AUS:
                    if frame.get(au, 0.0) > 0.2:
                        self.habitual_counts[au] += 1

    def analyze(
        self,
        current_aus: dict[str, float],
        timestamp: float,
        dominant_emotion: str = "neutral",
        camera_quality_score: float = 1.0,
        noise_state: 'NoiseState | None' = None,
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

        # No second smoothing here. The AUs arrive already EMA-smoothed by
        # ActionUnitAnalyzer (alpha 0.5); a second EMA (alpha 0.4) on top made
        # the offset test impossible to pass: after any peak the next three
        # frames averaged at least ~59% of it, and the test needs them to fall
        # below half. Real expressions could never be detected.
        smoothed_aus = dict(current_aus)

        # ── Noise filtering ──────────────────────────────────────────────
        if noise_state is not None:
            # Skip entirely during yawns, scratching, or forced blinks
            if noise_state.is_yawning or noise_state.is_scratching or noise_state.is_forced_blink:
                return None
            
            # During speech, only analyze upper-face AUs. The excluded AUs are
            # left out of the reading (not stored as 0): _au_series stops at
            # the first reading without them, so speech never creates the
            # artificial 0 → value → 0 steps that used to pass as peaks.
            if noise_state.upper_face_only:
                filtered = {k: v for k, v in smoothed_aus.items() 
                            if k in noise_state.filtered_aus_for_micro}
                smoothed_aus = filtered

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
        Core detection logic: find AUs that just completed an onset→apex→offset
        and that peaked together, then validate them as one event.
        """
        aus_to_track = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9",
                        "AU12", "AU14", "AU15", "AU17", "AU20",
                        "AU23", "AU24", "AU25", "AU26"]

        candidates: list[AUPeak] = []
        for au_name in aus_to_track:
            peaks = self._check_au_pattern(au_name, timestamp)
            if peaks:
                candidates.extend(peaks)

        if not candidates:
            return None

        # ── Group by co-occurrence ───────────────────────────────────
        # Anchor on the strongest fresh peak and keep only the AUs that peaked
        # together with it. Previously every AU with a peak anywhere in the
        # 3 s buffer was pooled into one "event".
        anchor = max(candidates, key=lambda p: p.amplitude)
        group: dict[str, AUPeak] = {}
        for peak in candidates:
            if abs(peak.peak_time - anchor.peak_time) <= self.cooccurrence_window_s:
                best = group.get(peak.au)
                if best is None or peak.amplitude > best.amplitude:
                    group[peak.au] = peak

        # ── Layer 1: Temporal Filter ─────────────────────────────────
        avg_duration_ms = float(np.mean([p.duration for p in group.values()]) * 1000)
        if avg_duration_ms < self.min_duration_ms or avg_duration_ms > self.max_duration_ms:
            return None

        # ── Gap check ────────────────────────────────────────────────
        if self._recent_detections and (timestamp - self._recent_detections[-1]) < self._min_detection_gap:
            return None

        # ── Layer 2: Context Filter ──────────────────────────────────
        # Every peak already exceeds the person's baseline + 2.5 std (see
        # _check_au_pattern); what is left is dropping habitual movements.
        activated_aus = [au for au in group if not self._is_habitual(au)]
        if not activated_aus:
            return None

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
            peak_values={au: group[au].peak_value for au in activated_aus},
        )

        is_contradictory = micro_contradicts_displayed(detected_emotion, dominant_emotion)

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
            for au, peak in group.items():
                self._consumed_peaks[au] = max(self._consumed_peaks.get(au, -np.inf), peak.peak_time)
            return event

        return None  # Below threshold, not surfaced

    def _au_series(self, au_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        The most recent uninterrupted run of readings for one AU.

        Stops at the first reading that lacks the AU (it was excluded during
        speech) or at a time gap (frames skipped for noise or a lost face).
        """
        times: list[float] = []
        values: list[float] = []
        prev_time = None
        for reading in reversed(self.au_buffer):
            value = reading.values.get(au_name)
            if value is None:
                break
            if prev_time is not None and (prev_time - reading.timestamp) > self.max_frame_gap_s:
                break
            times.append(reading.timestamp)
            values.append(float(value))
            prev_time = reading.timestamp
        return np.array(times[::-1]), np.array(values[::-1])

    def _au_threshold(self, au_name: str, values: np.ndarray) -> tuple[float, float]:
        """(resting value, peak threshold) for one AU."""
        if self.baseline_set:
            rest = self.baseline_aus.get(au_name, 0.0)
            variability = self.baseline_variability.get(au_name, 0.05)
            return rest, max(rest + variability * 2.5, 0.15)
        return float(np.percentile(values, 20)), 0.3  # Fallback before calibration

    def _check_au_pattern(self, au_name: str, current_time: float) -> list[AUPeak] | None:
        """
        Completed, not-yet-reported onset→apex→offset peaks of one AU.

        Measured on the deviation from the person's resting value: the onset
        is where the AU last sat at or below half of the peak's height above
        rest, the offset where it first falls back there. The duration is the
        time between both crossings (full width at half maximum), interpolated
        between frames. The old test compared the peak with the absolute value
        of the three frames right next to it, which are part of the rise and
        the fall themselves, so it failed for any expression that did not start
        from exactly zero.
        """
        times, values = self._au_series(au_name)
        if len(values) < 4:
            return None

        rest, threshold = self._au_threshold(au_name, values)
        peak_indices, _ = find_peaks(values, height=threshold)
        if len(peak_indices) == 0:
            return None

        already_reported = self._consumed_peaks.get(au_name, -np.inf)
        results: list[AUPeak] = []

        for peak_idx in peak_indices:
            if times[peak_idx] <= already_reported:
                continue

            peak_val = float(values[peak_idx])
            amplitude = peak_val - rest
            if amplitude <= 0:
                continue
            half = rest + amplitude / 2.0

            onset_idx = next((i for i in range(peak_idx - 1, -1, -1) if values[i] <= half), None)
            if onset_idx is None:
                continue  # rise started before the series: no onset seen
            offset_idx = next((i for i in range(peak_idx + 1, len(values)) if values[i] <= half), None)
            if offset_idx is None:
                continue  # has not returned to rest yet

            if current_time - times[offset_idx] > self.completion_window_s:
                continue  # completed too long ago; it was already evaluated

            # At least two frames above the threshold. A one-frame excursion is
            # landmark jitter as often as expression, and at ~20 fps a real
            # movement of 100 ms or more always spans two frames once smoothed.
            # This sets the practical minimum near 100 ms, below the 40 ms the
            # config nominally allows: one frame every 50 ms cannot tell a
            # 40 ms expression apart from noise.
            above = int(np.sum(values[onset_idx:offset_idx + 1] >= threshold))
            if above < self.min_frames_above_threshold:
                continue

            onset_time = _crossing_time(times, values, onset_idx, onset_idx + 1, half)
            offset_time = _crossing_time(times, values, offset_idx - 1, offset_idx, half)
            duration = offset_time - onset_time
            if duration > 0:
                results.append(AUPeak(
                    au=au_name,
                    peak_time=float(times[peak_idx]),
                    peak_value=peak_val,
                    amplitude=float(amplitude),
                    duration=float(duration),
                    offset_time=float(times[offset_idx]),
                ))

        return results or None

    def _is_habitual(self, au_name: str) -> bool:
        """Check if an AU activation is a habitual movement for this person."""
        if au_name in self.habitual_counts and self._calibration_frame_count > 0:
            ratio = self.habitual_counts[au_name] / self._calibration_frame_count
            return ratio >= self.habitual_ratio_threshold
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
        peak_values: dict[str, float] | None = None,
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
        if micro_contradicts_displayed(detected_emotion, dominant_emotion):
            score += 30  # Big boost for contradictory micro-expressions

        # Factor 4: Duration (modal range per Yan et al. 2013 is 80-200ms)
        if 80 <= duration_ms <= 200:
            score += 20  # Optimal micro-expression duration
        elif 40 <= duration_ms <= 500:
            score += 10  # Acceptable range

        # Factor 5: Baseline deviation (if available)
        if self.baseline_set:
            deviations = []
            for au in activated_aus:
                baseline_val = self.baseline_aus.get(au, 0.0)
                if peak_values and au in peak_values:
                    peak = peak_values[au]
                else:
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


def _crossing_time(times: np.ndarray, values: np.ndarray, i: int, j: int, level: float) -> float:
    """Time at which the segment between samples i and j crosses `level`."""
    v_i, v_j = float(values[i]), float(values[j])
    if v_j == v_i:
        return float(times[i])
    fraction = float(np.clip((level - v_i) / (v_j - v_i), 0.0, 1.0))
    return float(times[i] + fraction * (times[j] - times[i]))
