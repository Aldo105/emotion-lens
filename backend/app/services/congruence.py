"""
EmotionLens — Trustworthiness / Congruence Scoring Engine

Computes a congruence score (0-100) measuring how consistent a person's
facial expressions are with their apparent emotional state.

Score Components:
  1. Expression Stability (30%) — Consistency of emotions over time
  2. Micro-Expression Alignment (35%) — Do micro-expressions match the dominant emotion?
  3. Baseline Deviation (20%) — How much behavior deviates from calibration baseline
  4. Physiological Consistency (15%) — Blink rate, gaze stability, head movement

This is an analytical aid, NOT a lie detector. Results should supplement,
not replace, human judgment.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from backend.app.config import settings


@dataclass
class CongruenceBreakdown:
    """Detailed breakdown of the congruence score components."""
    stability_score: float       # 0-100
    micro_alignment_score: float # 0-100
    baseline_score: float        # 0-100
    physiological_score: float   # 0-100
    overall_score: float         # 0-100 (weighted average)


class CongruenceScorer:
    """
    Computes trustworthiness/congruence scores from emotion and AU data.
    
    Maintains a history of recent emotions and events to assess
    temporal consistency and alignment.
    """

    def __init__(self):
        # Weights (from config)
        self.w_stability = settings.congruence_weight_stability         # 0.30
        self.w_micro = settings.congruence_weight_micro_alignment       # 0.35
        self.w_baseline = settings.congruence_weight_baseline           # 0.20
        self.w_physio = settings.congruence_weight_physiological        # 0.15

        # Rolling emotion history (last 30 seconds)
        self.emotion_history: deque[dict] = deque(maxlen=300)  # ~10 fps * 30s

        # Micro-expression event log
        self.micro_events: list[dict] = []

        # Baseline reference
        self.baseline_aus: dict[str, float] = {}
        self.baseline_set = False

        # Smoothing — keep recent scores for smoothing output
        self._score_buffer: deque[float] = deque(maxlen=15)

    def set_baseline(self, baseline_aus: dict[str, float]):
        """Set the baseline AU values from calibration."""
        self.baseline_aus = baseline_aus
        self.baseline_set = True

    def add_micro_expression(self, event: dict):
        """Record a micro-expression event for alignment analysis."""
        self.micro_events.append(event)
        # Keep only last 50 events
        if len(self.micro_events) > 50:
            self.micro_events = self.micro_events[-50:]

    def compute(
        self,
        emotion: str,
        confidence: float,
        action_units: dict[str, float],
        timestamp: float,
    ) -> dict:
        """
        Compute the congruence score for the current frame.

        Args:
            emotion: Current detected emotion.
            confidence: Confidence of the emotion prediction.
            action_units: Current AU activations.
            timestamp: Current timestamp in seconds.

        Returns:
            {
                "score": float (0-100),
                "breakdown": {
                    "stability": float,
                    "micro_alignment": float,
                    "baseline": float,
                    "physiological": float,
                },
                "level": str ("high", "moderate", "low"),
                "color": str ("green", "yellow", "red"),
            }
        """
        # Record emotion in history
        self.emotion_history.append({
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": timestamp,
            "action_units": action_units.copy(),
        })

        # Compute each component
        stability = self._compute_stability()
        micro_alignment = self._compute_micro_alignment(emotion)
        baseline_dev = self._compute_baseline_deviation(action_units)
        physiological = self._compute_physiological(action_units)

        # Weighted average
        overall = (
            stability * self.w_stability +
            micro_alignment * self.w_micro +
            baseline_dev * self.w_baseline +
            physiological * self.w_physio
        )

        # Smooth the score to avoid jitter
        self._score_buffer.append(overall)
        smoothed_score = float(np.mean(self._score_buffer))

        # Determine level and color
        if smoothed_score >= 80:
            level, color = "high", "green"
        elif smoothed_score >= 50:
            level, color = "moderate", "yellow"
        else:
            level, color = "low", "red"

        return {
            "score": round(smoothed_score, 1),
            "breakdown": {
                "stability": round(stability, 1),
                "micro_alignment": round(micro_alignment, 1),
                "baseline": round(baseline_dev, 1),
                "physiological": round(physiological, 1),
            },
            "level": level,
            "color": color,
        }

    # ══════════════════════════════════════════════════════════════════
    # COMPONENT SCORERS
    # ══════════════════════════════════════════════════════════════════

    def _compute_stability(self) -> float:
        """
        Expression Stability (0-100):
        How consistent are emotions over time?
        Rapid, unexplained shifts = lower score.
        """
        if len(self.emotion_history) < 5:
            return 50.0  # Warmup default

        # Look at last 30 readings
        recent = list(self.emotion_history)[-30:]
        emotions = [r["emotion"] for r in recent]

        if not emotions:
            return 50.0

        # Count emotion transitions
        transitions = sum(
            1 for i in range(1, len(emotions))
            if emotions[i] != emotions[i - 1]
        )

        transition_rate = transitions / max(len(emotions) - 1, 1)

        # Low transition rate = high stability
        # 0 transitions = 100, >50% transitions = 20
        stability = 100.0 - (transition_rate * 100.0)
        return float(np.clip(stability, 20.0, 100.0))

    def _compute_micro_alignment(self, current_emotion: str) -> float:
        """
        Micro-Expression Alignment (0-100):
        Do recent micro-expressions match the dominant emotion?
        Contradictory micro-expressions = lower score.
        """
        if not self.micro_events:
            return 60.0  # No micro-expressions = baseline moderate default

        # Look at recent micro-expressions (last 10)
        recent_micros = self.micro_events[-10:]

        contradictory_count = 0
        aligned_count = 0
        total_relevance = 0

        for event in recent_micros:
            relevance = event.get("relevance_score", 50)
            total_relevance += relevance

            if event.get("is_contradictory", False):
                contradictory_count += 1
            else:
                aligned_count += 1

        total = len(recent_micros)
        if total == 0:
            return 60.0

        # More contradictory = lower alignment
        contradiction_ratio = contradictory_count / total
        
        # Weight by average relevance
        avg_relevance = total_relevance / total / 100.0  # normalize to 0-1

        # Score: 100 if no contradictions, drops with more contradictions
        alignment = 100.0 - (contradiction_ratio * 80.0 * avg_relevance)
        return float(np.clip(alignment, 10.0, 100.0))

    def _compute_baseline_deviation(self, current_aus: dict[str, float]) -> float:
        """
        Baseline Deviation (0-100):
        How much does current behavior deviate from the calibration baseline?
        Large deviations = lower score (possible concealment).
        """
        if not self.baseline_set or not self.baseline_aus:
            return 60.0  # Default when no baseline
 
        deviations = []
        for au_name, current_val in current_aus.items():
            if au_name in self.baseline_aus:
                baseline_val = self.baseline_aus[au_name]
                deviation = abs(current_val - baseline_val)
                deviations.append(deviation)
 
        if not deviations:
            return 60.0

        avg_deviation = np.mean(deviations)

        # Low deviation = high score, high deviation = low score
        # Scale: 0.0 deviation = 100, 0.5 deviation = 20
        score = 100.0 - (avg_deviation * 160.0)
        return float(np.clip(score, 20.0, 100.0))

    def _compute_physiological(self, action_units: dict[str, float]) -> float:
        """
        Physiological Consistency (0-100):
        Blink rate stability, gaze steadiness, head movement.
        Erratic patterns suggest stress/discomfort.
        """
        blink_rate = action_units.get("blink_rate", 0.5)
        gaze_stability = action_units.get("gaze_stability", 0.5)
        head_tilt = action_units.get("head_tilt", 0.0)
        hr_stress = action_units.get("hr_stress")

        # Blink rate: normal (0.3-0.5) = high score, extreme = low
        if blink_rate < 0.3:
            blink_score = 80.0  # Normal low blink rate
        elif blink_rate < 0.6:
            blink_score = 90.0  # Normal range
        elif blink_rate < 0.8:
            blink_score = 60.0  # Elevated
        else:
            blink_score = 30.0  # Very elevated (nervousness)

        # Gaze: high stability = high score
        gaze_score = gaze_stability * 100.0

        # Head tilt: low = stable = high score
        head_score = (1.0 - head_tilt) * 100.0

        # Weighted combination
        if hr_stress is not None:
            # hr_stress: 0.0 (no stress) to 1.0 (stressed)
            hr_stress_score = (1.0 - hr_stress) * 100.0
            physio = (
                blink_score * 0.25 +
                gaze_score * 0.35 +
                head_score * 0.15 +
                hr_stress_score * 0.25
            )
        else:
            physio = blink_score * 0.35 + gaze_score * 0.45 + head_score * 0.20

        return float(np.clip(physio, 10.0, 100.0))
