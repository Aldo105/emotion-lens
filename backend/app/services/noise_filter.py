"""
EmotionLens — Facial Noise Filter Service

Filters facial movements that are NOT emotional expressions, such as speaking,
yawning, scratching, tics, and forced blinks.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


LOWER_FACE_AUS = frozenset({
    "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
})


@dataclass
class NoiseState:
    is_speaking: bool
    is_yawning: bool
    is_tic: bool
    is_scratching: bool
    is_forced_blink: bool
    noise_type: str | None
    upper_face_only: bool
    filtered_aus_for_micro: list[str]


class FacialNoiseFilter:
    """
    Filters facial movements that are NOT emotional expressions.
    
    Detects and marks:
    - Active speech (mouth AUs consistently active = person is talking)
    - Yawns (AU25+AU26 high + long duration + no emotional eye component)
    - Recurring nervous tics (same AU pattern repeating >3 times)
    - Scratching/itching (asymmetric unilateral mouth movement without eye activation)
    - Forced blinks (AU45 prolonged >300ms vs normal blink ~150ms)
    """

    def __init__(self):
        # Speech tracking
        self.speech_buffer: deque[tuple[float, bool]] = deque()
        
        # Yawn tracking
        self.yawn_start_time: float | None = None
        self._yawning_last_frame = False
        
        # Blink tracking
        self.blink_start_time: float | None = None
        self._forced_blink_last_frame = False
        
        # Tic tracking
        self.au_states: dict[str, bool] = {}
        self.au_spikes: dict[str, list[float]] = {}
        # Recent values per AU (~10 s at 20 fps). Live sessions have no
        # calibration, and judging spikes against 0 flagged a "tic" on most
        # frames because several AUs sit near 1.0 at rest.
        self._au_history: dict[str, deque[float]] = {}
        
        self.baseline_aus: dict[str, float] = {}
        
        # Stats
        self.stats = {
            "speech_frames": 0,
            "yawn_events": 0,
            "tic_events": 0,
            "scratch_frames": 0,
            "forced_blink_events": 0,
        }

    def set_baseline(self, baseline_aus: dict[str, float]) -> None:
        """Calibrate thresholds based on baseline AU data."""
        self.baseline_aus = baseline_aus.copy()

    def analyze(self, current_aus: dict[str, float], timestamp: float) -> NoiseState:
        """
        Analyze current AUs for noise patterns.
        
        Args:
            current_aus: Dict mapping AU names to activation levels (0.0 - 1.0).
            timestamp: Current timestamp in seconds.
            
        Returns:
            NoiseState object with boolean flags and filtered AU list.
        """
        is_speaking = self._detect_speech(current_aus, timestamp)
        is_yawning = self._detect_yawn(current_aus, timestamp)
        is_tic = self._detect_tic(current_aus, timestamp)
        is_scratching = self._detect_scratch(current_aus)
        is_forced_blink = self._detect_forced_blink(current_aus, timestamp)

        # Determine overall noise type and update stats
        noise_type = None
        if is_yawning:
            noise_type = "yawn"
            if not self._yawning_last_frame:
                self.stats["yawn_events"] += 1
        elif is_forced_blink:
            noise_type = "forced_blink"
            if not self._forced_blink_last_frame:
                self.stats["forced_blink_events"] += 1
        elif is_tic:
            noise_type = "tic"
        elif is_scratching:
            noise_type = "scratch"
            self.stats["scratch_frames"] += 1
        elif is_speaking:
            noise_type = "speech"
            self.stats["speech_frames"] += 1

        self._yawning_last_frame = is_yawning
        self._forced_blink_last_frame = is_forced_blink

        # Determine which AUs to keep for micro-expression analysis
        all_aus = [k for k in current_aus.keys() if k.startswith("AU")]
        filtered_aus = all_aus.copy()
        upper_face_only = False

        if is_speaking:
            upper_face_only = True
            allowed_aus = {"AU1", "AU2", "AU4", "AU7", "AU9"}
            filtered_aus = [au for au in filtered_aus if au in allowed_aus]
        elif is_yawning or is_scratching:
            # During a yawn or scratch, typically skip micro-expression analysis
            filtered_aus = []

        return NoiseState(
            is_speaking=is_speaking,
            is_yawning=is_yawning,
            is_tic=is_tic,
            is_scratching=is_scratching,
            is_forced_blink=is_forced_blink,
            noise_type=noise_type,
            upper_face_only=upper_face_only,
            filtered_aus_for_micro=filtered_aus
        )
        
    def _detect_speech(self, aus: dict[str, float], timestamp: float) -> bool:
        """Active speech (mouth AUs consistently active over 3s window)."""
        au25 = aus.get("AU25", 0.0)
        au26 = aus.get("AU26", 0.0)
        au12 = aus.get("AU12", 0.0)  # Lip corner puller (smile indicator)

        # (Improvement 9): Don't flag as speech when AU12 is active —
        # that's a smile/laugh, not speech. Previously, smiling with teeth
        # showing triggered false speech detection, disabling lower-face AUs.
        is_smiling = au12 > 0.3
        is_active = (au25 > 0.2 or au26 > 0.2) and not is_smiling

        self.speech_buffer.append((timestamp, is_active))
        
        # Remove readings older than 3 seconds
        while self.speech_buffer and timestamp - self.speech_buffer[0][0] > 3.0:
            self.speech_buffer.popleft()
            
        if not self.speech_buffer:
            return False
            
        active_count = sum(1 for _, active in self.speech_buffer if active)
        ratio = active_count / len(self.speech_buffer)
        
        return ratio > 0.6
        
    def _detect_yawn(self, aus: dict[str, float], timestamp: float) -> bool:
        """Yawns (AU25+AU26 high + long duration + no emotional eye component)."""
        au25 = aus.get("AU25", 0.0)
        au26 = aus.get("AU26", 0.0)
        au6 = aus.get("AU6", 0.0)
        au1 = aus.get("AU1", 0.0)
        
        if au25 > 0.7 and au26 > 0.6 and au6 < 0.2 and au1 < 0.2:
            if self.yawn_start_time is None:
                self.yawn_start_time = timestamp
            elif timestamp - self.yawn_start_time > 2.0:
                return True
        else:
            self.yawn_start_time = None
            
        return False
        
    def _detect_tic(self, aus: dict[str, float], timestamp: float) -> bool:
        """Recurring nervous tics (same AU pattern repeating >3 times within 10s)."""
        is_tic = False
        # Blinks repeat several times every 10 s on their own (AU5 rises with
        # them); they are tracked as blink rate, not as tics.
        aus_to_check = [k for k in aus.keys() if k.startswith("AU") and k not in ("AU45", "AU5")]
        # A mouth that keeps opening and closing is talking; its lower-face
        # AUs swing with every syllable. The speech detector above misses it
        # (AU12 rests near 0.4 here, which it reads as a smile), so the swing
        # of AU25 itself is used to keep those AUs out of the tic count.
        lips = self._au_history.get("AU25")
        if lips and len(lips) >= 20:
            lo, hi = np.percentile(np.fromiter(lips, dtype=float), [10, 90])
            if hi - lo > 0.5:
                aus_to_check = [k for k in aus_to_check if k not in LOWER_FACE_AUS]
        
        for au in aus_to_check:
            val = aus.get(au, 0.0)
            if au in self.baseline_aus:
                threshold = self.baseline_aus[au] + 0.3
            else:
                # No calibration: the AU's own recent median, and a margin that
                # grows with how much it normally moves (talking keeps the mouth
                # AUs swinging, which is not a repeated involuntary movement).
                history = self._au_history.get(au)
                if history:
                    values = np.fromiter(history, dtype=float)
                    median = float(np.median(values))
                    spread = 1.4826 * float(np.median(np.abs(values - median)))
                    threshold = median + max(0.3, 3.0 * spread)
                else:
                    threshold = val + 0.3
            
            currently_active = val > threshold
            was_active = self.au_states.get(au, False)
            
            if was_active and not currently_active:
                # Spike completed (went from above threshold to below)
                if au not in self.au_spikes:
                    self.au_spikes[au] = []
                self.au_spikes[au].append(timestamp)
                
                # Check how many spikes in the last 10 seconds
                recent_spikes = [t for t in self.au_spikes[au] if timestamp - t <= 10.0]
                if len(recent_spikes) == 4:
                    self.stats["tic_events"] += 1
                
            self.au_states[au] = currently_active
            
            # Clean old spikes and evaluate if currently in tic state
            if au in self.au_spikes:
                self.au_spikes[au] = [t for t in self.au_spikes[au] if timestamp - t <= 10.0]
                if len(self.au_spikes[au]) > 3:
                    is_tic = True

        # After the checks, so each frame is judged against the ones before it.
        for au, val in aus.items():
            if au.startswith("AU"):
                self._au_history.setdefault(au, deque(maxlen=200)).append(val)
                    
        return is_tic
        
    def _detect_scratch(self, aus: dict[str, float]) -> bool:
        """Scratching/itching (asymmetric unilateral mouth movement without eye activation)."""
        face_symmetry = aus.get("face_symmetry", 1.0)
        
        lower_aus = ["AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26"]
        upper_aus = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU9", "AU45"]
        
        max_lower = max([aus.get(au, 0.0) for au in lower_aus]) if lower_aus else 0.0
        max_upper = max([aus.get(au, 0.0) for au in upper_aus]) if upper_aus else 0.0
        
        # Asymmetry indicator + strong lower face movement + weak upper face movement
        if face_symmetry < 0.7 and max_lower > 0.4 and max_upper < 0.2:
            return True
            
        return False
        
    def _detect_forced_blink(self, aus: dict[str, float], timestamp: float) -> bool:
        """Forced blinks (AU45 prolonged >300ms)."""
        au45 = aus.get("AU45", 0.0)
        if au45 > 0.5:
            if self.blink_start_time is None:
                self.blink_start_time = timestamp
            elif timestamp - self.blink_start_time > 0.3:
                return True
        else:
            self.blink_start_time = None
            
        return False

    def get_session_stats(self) -> dict:
        """Return total counts of each noise type for the session."""
        return self.stats
