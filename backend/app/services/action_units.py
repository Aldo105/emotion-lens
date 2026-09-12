"""
EmotionLens — FACS Action Unit Analyzer

Computes Facial Action Coding System (FACS) Action Units from MediaPipe
468-point face mesh landmarks. Uses geometric distances and ratios between
specific landmark pairs to estimate muscle activation levels.

Each AU maps to specific facial muscles. The activation level (0.0 - 1.0)
represents how strongly that muscle group is engaged.

Reference: Ekman & Friesen, Facial Action Coding System (1978)
"""

import numpy as np

from backend.app.services.face_detector import LANDMARK_GROUPS


class ActionUnitAnalyzer:
    """
    Computes FACS Action Unit activation levels from facial landmarks.
    
    All computations are pure geometry — no ML model needed.
    Distances are normalized by inter-ocular distance to be scale-invariant.
    """

    def __init__(self):
        # Calibration baselines (set during baseline period)
        self.baseline_aus = None
        self.baseline_variability = None  # Per-AU standard deviation
        self.baseline_set = False
        self._baseline_buffer = []  # Collects AU readings during calibration

        # Blink tracking
        self._blink_timestamps = []
        self._last_eye_ratio = None

        # EMA smoothing (Fase 4.2)
        self._prev_smoothed_aus = None
        self._ema_alpha = 0.5

    def compute(
        self,
        landmarks: list[dict],
        frame_shape: tuple[int, int],
        timestamp: float = 0.0,
    ) -> dict[str, float]:
        """
        Compute all Action Unit activations from landmarks.

        Args:
            landmarks: List of 468 landmarks with 'x', 'y', 'z' keys (normalized 0-1).
            frame_shape: (height, width) of the frame.
            timestamp: Current timestamp in seconds (for blink rate tracking).

        Returns:
            Dict mapping AU names to activation levels (0.0 - 1.0).
            Also includes derived metrics like 'blink_rate' and 'gaze_stability'.
        """
        if not landmarks or len(landmarks) < 468:
            return {}

        # Convert to numpy array for vectorized math
        lm = np.array([[l["x"], l["y"], l["z"]] for l in landmarks], dtype=np.float32)
        h, w = frame_shape

        # ── Head Pose Compensation (Fase 4.1) ──
        yaw, pitch = self._estimate_yaw_pitch(landmarks)
        if abs(yaw) > 0.05 or abs(pitch) > 0.05:
            lm = self._compensate_head_pose(lm, yaw, pitch)

        # Normalize by inter-ocular distance (scale-invariant)
        iod = self._inter_ocular_distance(lm)
        if iod < 1e-6:
            return {}

        aus = {}

        # ── Brow Action Units ────────────────────────────────────────
        aus["AU1"] = self._au1_inner_brow_raiser(lm, iod)
        aus["AU2"] = self._au2_outer_brow_raiser(lm, iod)
        aus["AU4"] = self._au4_brow_lowerer(lm, iod)

        # ── Eye Action Units ─────────────────────────────────────────
        aus["AU6"] = self._au6_cheek_raiser(lm, iod)
        aus["AU7"] = self._au7_lid_tightener(lm, iod)
        eye_ratio = self._eye_aspect_ratio(lm)
        aus["AU45"] = 1.0 if eye_ratio < 0.15 else 0.0  # Blink

        # ── Nose Action Units ────────────────────────────────────────
        aus["AU9"] = self._au9_nose_wrinkler(lm, iod)

        # ── Mouth / Lip Action Units ─────────────────────────────────
        aus["AU12"] = self._au12_lip_corner_puller(lm, iod)
        aus["AU14"] = self._au14_dimpler(lm, iod)
        aus["AU15"] = self._au15_lip_corner_depressor(lm, iod)
        aus["AU17"] = self._au17_chin_raiser(lm, iod)
        aus["AU20"] = self._au20_lip_stretcher(lm, iod)
        aus["AU23"] = self._au23_lip_tightener(lm, iod)
        aus["AU24"] = self._au24_lip_pressor(lm, iod)
        aus["AU25"] = self._au25_lips_part(lm, iod)
        aus["AU26"] = self._au26_jaw_drop(lm, iod)

        # ── Derived Metrics ──────────────────────────────────────────
        aus["blink_rate"] = self._compute_blink_rate(aus["AU45"], timestamp)
        aus["gaze_stability"] = self._compute_gaze_stability(lm)
        aus["head_tilt"] = self._compute_head_tilt(lm)
        aus["face_symmetry"] = self._compute_face_symmetry(lm, iod)

        # Clamp all values to [0.0, 1.0]
        aus = {k: float(np.clip(v, 0.0, 1.0)) for k, v in aus.items()}

        # Apply EMA smoothing (Fase 4.2)
        if self._prev_smoothed_aus is None:
            self._prev_smoothed_aus = aus.copy()
        else:
            smoothed = {}
            for k, v in aus.items():
                if k in ("blink_rate", "gaze_stability", "head_tilt", "face_symmetry"):
                    smoothed[k] = v
                else:
                    prev = self._prev_smoothed_aus.get(k, v)
                    smoothed[k] = self._ema_alpha * v + (1 - self._ema_alpha) * prev
            self._prev_smoothed_aus = smoothed.copy()
            aus = smoothed.copy()

        # Store for baseline calibration
        if not self.baseline_set:
            self._baseline_buffer.append(aus.copy())

        return aus

    def _estimate_yaw_pitch(self, landmarks: list) -> tuple[float, float]:
        """Estimate yaw and pitch in radians from raw landmarks."""
        try:
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            forehead = landmarks[10]
            chin = landmarks[152]

            eye_mid_x = (left_eye["x"] + right_eye["x"]) / 2.0
            eye_width = abs(right_eye["x"] - left_eye["x"])
            if eye_width > 0.001:
                yaw_ratio = (nose["x"] - eye_mid_x) / eye_width
                yaw = yaw_ratio * 1.0  # rad
            else:
                yaw = 0.0

            vert_mid_y = (forehead["y"] + chin["y"]) / 2.0
            vert_span = abs(chin["y"] - forehead["y"])
            if vert_span > 0.001:
                pitch_ratio = (nose["y"] - vert_mid_y) / vert_span
                pitch = pitch_ratio * 1.0  # rad
            else:
                pitch = 0.0

            return yaw, pitch
        except Exception:
            return 0.0, 0.0

    def _compensate_head_pose(self, lm: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """Rotate landmarks around nose tip to compensate head pose orientation."""
        try:
            # Rotation around Y axis (yaw)
            cy, sy = np.cos(-yaw), np.sin(-yaw)
            R_y = np.array([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]
            ], dtype=np.float32)

            # Rotation around X axis (pitch)
            cp, sp = np.cos(-pitch), np.sin(-pitch)
            R_x = np.array([
                [1, 0, 0],
                [0, cp, -sp],
                [0, sp, cp]
            ], dtype=np.float32)

            R = R_y @ R_x
            nose_center = lm[1].copy()
            centered = lm - nose_center
            rotated = centered @ R.T
            return rotated + nose_center
        except Exception:
            return lm

    def set_baseline(self):
        """
        Set the baseline from accumulated readings.
        Called after the calibration period (typically 30 seconds).
        
        Computes both the mean AND the standard deviation per AU.
        The standard deviation is critical for adaptive micro-expression
        thresholds — each person has different natural facial variability.
        """
        if not self._baseline_buffer:
            return

        # Average and std for all AU readings to get stable baseline
        self.baseline_aus = {}
        self.baseline_variability = {}
        all_keys = self._baseline_buffer[0].keys()

        for key in all_keys:
            values = [reading[key] for reading in self._baseline_buffer if key in reading]
            self.baseline_aus[key] = float(np.mean(values))
            self.baseline_variability[key] = float(np.std(values)) if len(values) > 1 else 0.05

        self.baseline_set = True
        n_frames = len(self._baseline_buffer)
        self._baseline_buffer.clear()
        print(f"[OK] Baseline calibrated with {len(self.baseline_aus)} AUs from {n_frames} frames")
        print(f"[OK] Per-AU variability range: "
              f"min={min(self.baseline_variability.values()):.4f}, "
              f"max={max(self.baseline_variability.values()):.4f}, "
              f"mean={np.mean(list(self.baseline_variability.values())):.4f}")

    def get_deviation_from_baseline(self, current_aus: dict[str, float]) -> dict[str, float]:
        """
        Compute how much current AU activations deviate from the baseline.
        Positive = more activated than baseline, Negative = less.
        """
        if not self.baseline_set or not self.baseline_aus:
            return {k: 0.0 for k in current_aus}

        deviations = {}
        for key, value in current_aus.items():
            baseline_val = self.baseline_aus.get(key, 0.0)
            deviations[key] = value - baseline_val

        return deviations

    # ══════════════════════════════════════════════════════════════════
    # INDIVIDUAL AU COMPUTATIONS
    # ══════════════════════════════════════════════════════════════════

    def _inter_ocular_distance(self, lm: np.ndarray) -> float:
        """Distance between inner corners of the eyes (normalization reference)."""
        left_inner = lm[133]   # Left eye inner corner
        right_inner = lm[362]  # Right eye inner corner
        return float(np.linalg.norm(left_inner[:2] - right_inner[:2]))

    def _dist(self, lm: np.ndarray, i: int, j: int) -> float:
        """Euclidean distance between two landmarks (2D)."""
        return float(np.linalg.norm(lm[i][:2] - lm[j][:2]))

    # ── AU1: Inner Brow Raiser ───────────────────────────────────────
    def _au1_inner_brow_raiser(self, lm: np.ndarray, iod: float) -> float:
        """
        Measures upward movement of inner eyebrows.
        Computed as distance from inner brow to nose bridge, normalized.
        Higher = brows raised more.
        """
        # Distance from inner brows to eye level
        left_brow_to_eye = self._dist(lm, 107, 133)   # inner brow to inner eye
        right_brow_to_eye = self._dist(lm, 336, 362)
        avg = (left_brow_to_eye + right_brow_to_eye) / 2.0
        # Normalize and scale
        ratio = avg / iod
        return float(np.clip((ratio - 0.12) * 8.0, 0.0, 1.0))

    # ── AU2: Outer Brow Raiser ───────────────────────────────────────
    def _au2_outer_brow_raiser(self, lm: np.ndarray, iod: float) -> float:
        """Measures upward movement of outer eyebrows."""
        left = self._dist(lm, 70, 33)    # outer brow to outer eye corner
        right = self._dist(lm, 300, 263)
        avg = (left + right) / 2.0
        ratio = avg / iod
        return float(np.clip((ratio - 0.12) * 8.0, 0.0, 1.0))

    # ── AU4: Brow Lowerer ────────────────────────────────────────────
    def _au4_brow_lowerer(self, lm: np.ndarray, iod: float) -> float:
        """
        Measures downward + inward brow movement (furrowing).
        Inverse of AU1 — closer brows to eyes = more furrowing.
        """
        left_brow_to_eye = self._dist(lm, 107, 133)
        right_brow_to_eye = self._dist(lm, 336, 362)
        avg = (left_brow_to_eye + right_brow_to_eye) / 2.0
        ratio = avg / iod
        # Lower ratio = brows are closer to eyes = more furrowing
        return float(np.clip((0.16 - ratio) * 10.0, 0.0, 1.0))

    # ── AU6: Cheek Raiser ────────────────────────────────────────────
    def _au6_cheek_raiser(self, lm: np.ndarray, iod: float) -> float:
        """
        Measures cheek raising (genuine/Duchenne smile indicator).
        Detects narrowing of eyes from below (lower lid pushes up).
        """
        # Lower eyelid to iris/eye center distance
        left_lower = self._dist(lm, 145, 159)   # bottom to top of left eye
        right_lower = self._dist(lm, 374, 386)   # bottom to top of right eye
        avg_opening = (left_lower + right_lower) / 2.0
        ratio = avg_opening / iod
        # Smaller opening with AU12 active = cheek raiser
        return float(np.clip((0.08 - ratio) * 15.0, 0.0, 1.0))

    # ── AU7: Lid Tightener ───────────────────────────────────────────
    def _au7_lid_tightener(self, lm: np.ndarray, iod: float) -> float:
        """Measures tightening of eyelids (narrowing eyes)."""
        left_opening = self._dist(lm, 145, 159)
        right_opening = self._dist(lm, 374, 386)
        avg = (left_opening + right_opening) / 2.0
        ratio = avg / iod
        return float(np.clip((0.06 - ratio) * 20.0, 0.0, 1.0))

    # ── AU9: Nose Wrinkler ───────────────────────────────────────────
    def _au9_nose_wrinkler(self, lm: np.ndarray, iod: float) -> float:
        """
        Measures nose wrinkling (disgust indicator).
        Detected by upward movement of nose wing landmarks relative to nose tip.
        """
        left_wing_to_bridge = self._dist(lm, 49, 6)
        right_wing_to_bridge = self._dist(lm, 279, 6)
        avg = (left_wing_to_bridge + right_wing_to_bridge) / 2.0
        ratio = avg / iod
        return float(np.clip((0.15 - ratio) * 8.0, 0.0, 1.0))

    # ── AU12: Lip Corner Puller (Smile) ──────────────────────────────
    def _au12_lip_corner_puller(self, lm: np.ndarray, iod: float) -> float:
        """
        Measures outward + upward pull of lip corners.
        The primary smile action unit.
        """
        mouth_width = self._dist(lm, 61, 291)
        # Also check lip corner height relative to lip center
        left_corner_y = lm[61][1]
        right_corner_y = lm[291][1]
        lip_center_y = lm[13][1]
        # Corners higher than center = smile (in image coords, lower y = higher)
        corner_lift = lip_center_y - ((left_corner_y + right_corner_y) / 2.0)

        width_ratio = mouth_width / iod
        lift_score = float(np.clip(corner_lift * 30.0, 0.0, 1.0))
        width_score = float(np.clip((width_ratio - 0.25) * 5.0, 0.0, 1.0))

        return (lift_score * 0.6 + width_score * 0.4)

    # ── AU14: Dimpler ────────────────────────────────────────────────
    def _au14_dimpler(self, lm: np.ndarray, iod: float) -> float:
        """Measures inward pull at lip corners (nervousness indicator)."""
        # Lip corners pull inward (narrower mouth with tension)
        mouth_width = self._dist(lm, 61, 291)
        ratio = mouth_width / iod
        # Narrow mouth with lip tension
        return float(np.clip((0.25 - ratio) * 6.0, 0.0, 1.0))

    # ── AU15: Lip Corner Depressor ───────────────────────────────────
    def _au15_lip_corner_depressor(self, lm: np.ndarray, iod: float) -> float:
        """Measures downward pull of lip corners (sadness/frown)."""
        left_corner_y = lm[61][1]
        right_corner_y = lm[291][1]
        lip_center_y = lm[13][1]
        # Corners lower than center = frown
        corner_drop = ((left_corner_y + right_corner_y) / 2.0) - lip_center_y
        return float(np.clip(corner_drop * 30.0, 0.0, 1.0))

    # ── AU17: Chin Raiser ────────────────────────────────────────────
    def _au17_chin_raiser(self, lm: np.ndarray, iod: float) -> float:
        """Measures chin boss pushing up (doubt/contempt indicator)."""
        chin_to_lower_lip = self._dist(lm, 152, 17)
        ratio = chin_to_lower_lip / iod
        return float(np.clip((0.15 - ratio) * 10.0, 0.0, 1.0))

    # ── AU20: Lip Stretcher ──────────────────────────────────────────
    def _au20_lip_stretcher(self, lm: np.ndarray, iod: float) -> float:
        """Measures lateral lip stretching (fear indicator)."""
        mouth_width = self._dist(lm, 61, 291)
        mouth_height = self._dist(lm, 13, 14)
        # Wide mouth with small opening = stretching
        if mouth_height < 1e-6:
            return 0.0
        ratio = mouth_width / mouth_height
        return float(np.clip((ratio - 3.0) * 0.3, 0.0, 1.0))

    # ── AU23: Lip Tightener ──────────────────────────────────────────
    def _au23_lip_tightener(self, lm: np.ndarray, iod: float) -> float:
        """Measures lip pressing/tightening (anger suppression)."""
        lip_gap = self._dist(lm, 13, 14)
        ratio = lip_gap / iod
        # Very small lip gap = tightened
        return float(np.clip((0.02 - ratio) * 50.0, 0.0, 1.0))

    # ── AU24: Lip Pressor ────────────────────────────────────────────
    def _au24_lip_pressor(self, lm: np.ndarray, iod: float) -> float:
        """Measures lip compression (stress/tension indicator)."""
        lip_gap = self._dist(lm, 13, 14)
        mouth_width = self._dist(lm, 61, 291)
        ratio = lip_gap / iod
        width_ratio = mouth_width / iod
        # Compressed lips = small gap + moderate width
        compression = float(np.clip((0.03 - ratio) * 40.0, 0.0, 1.0))
        return compression

    # ── AU25: Lips Part ──────────────────────────────────────────────
    def _au25_lips_part(self, lm: np.ndarray, iod: float) -> float:
        """Measures lip parting (surprise component)."""
        lip_gap = self._dist(lm, 13, 14)
        ratio = lip_gap / iod
        return float(np.clip((ratio - 0.02) * 15.0, 0.0, 1.0))

    # ── AU26: Jaw Drop ───────────────────────────────────────────────
    def _au26_jaw_drop(self, lm: np.ndarray, iod: float) -> float:
        """Measures jaw opening (surprise/shock indicator)."""
        jaw_opening = self._dist(lm, 13, 17)  # upper lip to chin bottom
        ratio = jaw_opening / iod
        return float(np.clip((ratio - 0.08) * 6.0, 0.0, 1.0))

    # ══════════════════════════════════════════════════════════════════
    # DERIVED METRICS
    # ══════════════════════════════════════════════════════════════════

    def _eye_aspect_ratio(self, lm: np.ndarray) -> float:
        """Eye aspect ratio (EAR) for blink detection."""
        left_v1 = self._dist(lm, 159, 145)  # top-bottom (left eye)
        left_h = self._dist(lm, 33, 133)    # left-right (left eye)
        right_v1 = self._dist(lm, 386, 374)
        right_h = self._dist(lm, 263, 362)

        if left_h < 1e-6 or right_h < 1e-6:
            return 0.3  # Default open

        left_ear = left_v1 / left_h
        right_ear = right_v1 / right_h
        return (left_ear + right_ear) / 2.0

    def _compute_blink_rate(self, blink_au: float, timestamp: float) -> float:
        """
        Track blinks and compute blink rate (blinks per minute).
        Normal: 15-20/min. Elevated: nervousness indicator.
        """
        if blink_au > 0.5:  # Blink detected
            if not self._blink_timestamps or (timestamp - self._blink_timestamps[-1]) > 0.1:
                self._blink_timestamps.append(timestamp)

        # Keep only last 60 seconds of blinks
        cutoff = timestamp - 60.0
        self._blink_timestamps = [t for t in self._blink_timestamps if t > cutoff]

        # Calculate rate
        if timestamp < 10:  # Need at least 10 seconds of data
            return 0.5  # Default to neutral

        blinks_per_min = len(self._blink_timestamps) * (60.0 / max(timestamp, 1.0))

        # Normalize: 15-20 bpm = normal (0.3-0.5), >30 = high nervousness (1.0)
        return float(np.clip((blinks_per_min - 15) / 25.0, 0.0, 1.0))

    def _compute_gaze_stability(self, lm: np.ndarray) -> float:
        """
        Estimate gaze stability from iris position relative to eye corners.
        Stable gaze = high confidence indicator.
        """
        if len(lm) <= 473:  # No iris landmarks
            return 0.5  # Default to neutral

        # Left iris center relative to left eye
        left_iris = lm[468][:2]
        left_inner = lm[133][:2]
        left_outer = lm[33][:2]
        left_center = (left_inner + left_outer) / 2.0
        left_deviation = np.linalg.norm(left_iris - left_center)

        # Right iris center relative to right eye
        right_iris = lm[473][:2]
        right_inner = lm[362][:2]
        right_outer = lm[263][:2]
        right_center = (right_inner + right_outer) / 2.0
        right_deviation = np.linalg.norm(right_iris - right_center)

        avg_deviation = (left_deviation + right_deviation) / 2.0
        eye_width = np.linalg.norm(left_outer - left_inner)

        if eye_width < 1e-6:
            return 0.5

        # Low deviation = centered gaze = high stability
        stability = float(1.0 - np.clip(avg_deviation / (eye_width * 0.3), 0.0, 1.0))
        return stability

    def _compute_head_tilt(self, lm: np.ndarray) -> float:
        """Compute head tilt angle (0 = straight, 1 = significant tilt)."""
        left_eye = lm[33][:2]
        right_eye = lm[263][:2]
        diff = right_eye - left_eye
        angle = np.abs(np.arctan2(diff[1], diff[0]))
        # Normalize: 0 degrees = 0, 15+ degrees = 1.0
        return float(np.clip(angle / 0.26, 0.0, 1.0))  # 0.26 rad ~ 15 degrees

    def _compute_face_symmetry(self, lm: np.ndarray, iod: float) -> float:
        """
        Compute facial symmetry (0 = asymmetric, 1 = symmetric).
        Asymmetry can indicate micro-expressions or contempt.
        """
        # Compare left vs right lip corner heights
        left_corner_y = lm[61][1]
        right_corner_y = lm[291][1]
        lip_asymmetry = abs(left_corner_y - right_corner_y) / iod

        # Compare left vs right brow heights
        left_brow_y = lm[107][1]
        right_brow_y = lm[336][1]
        brow_asymmetry = abs(left_brow_y - right_brow_y) / iod

        total_asymmetry = (lip_asymmetry + brow_asymmetry) / 2.0
        symmetry = 1.0 - float(np.clip(total_asymmetry * 15.0, 0.0, 1.0))
        return symmetry
