"""
EmotionLens — Eulerian Video Magnification & Heart Rate Estimator

Integrates Eulerian Color Magnification to detect subtle skin color changes
caused by blood flow, enabling remote heart rate (rPPG) estimation.

Algorithm Pipeline:
  1. Extract forehead ROI using MediaPipe landmarks
  2. Extract mean R, G, B values from the forehead ROI
  3. Extract nose-bridge reference ROI and subtract common-mode noise
  4. Build signal buffers (rolling window of ~10 seconds)
  5. Compute CHROM pulse signal (de Haan & Jeanne, 2013)
  6. Apply Butterworth bandpass filter (0.7 - 4.0 Hz = 42-240 BPM)
  7. FFT to find dominant frequency -> convert to BPM
  8. Motion artifact rejection via landmark velocity
  9. Signal quality (SNR) computation
  10. Optionally: amplify color changes for visual feedback

Based on MIT CSAIL's "Eulerian Video Magnification for Revealing
Subtle Changes in the World" (Wu et al., 2012).

CHROM method based on: de Haan, G. & Jeanne, V. (2013).
"Robust Pulse Rate From Chrominance-Based rPPG."
IEEE Transactions on Biomedical Engineering.

Reference: https://github.com/joeljose/Eulerian-Video-Magnification
"""

from collections import deque

import cv2
import numpy as np

try:
    from scipy.signal import butter, detrend, filtfilt
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[!!] scipy not installed. Heart rate estimation will not work.")


class HeartRateEstimator:
    """
    Estimates heart rate from facial video using remote photoplethysmography (rPPG).
    
    Uses the forehead region (most stable for pulse detection) extracted via
    MediaPipe landmarks. Applies the CHROM (Chrominance-based) algorithm
    on R, G, B channels to isolate the pulse signal, with a nose-bridge
    reference ROI for common-mode noise removal and motion artifact rejection
    via landmark velocity tracking.
    """

    def __init__(
        self,
        buffer_seconds: float = 12.0,
        fps: float = 15.0,
        freq_low: float = 0.7,    # 42 BPM minimum
        freq_high: float = 2.5,   # 150 BPM maximum
        amplification_factor: float = 50.0,
        motion_threshold: float = 15.0,  # Normalized pixel displacement threshold
        min_estimation_seconds: float = 8.0,
        min_confidence: float = 0.40,
        max_bpm_change_per_second: float = 8.0,
        stale_after_seconds: float = 5.0,
        max_raw_spread_bpm: float = 12.0,
        estimation_interval_seconds: float = 0.5,
    ):
        """
        Args:
            buffer_seconds: How many seconds of signal to keep for analysis.
            fps: Expected frames per second (for frequency calculations).
            freq_low: Low cutoff frequency in Hz (0.7 = 42 BPM).
            freq_high: High cutoff frequency in Hz (2.5 = 150 BPM). Deliberately
                       narrower than the 4.0 Hz (240 BPM) rPPG literature ceiling:
                       240 BPM is unreachable for a seated subject, and every
                       extra Hz of band is extra room for a noise peak to win the
                       argmax. Widening this reintroduces readings of 190-220 BPM
                       under webcam noise (see references.py, "Banda del
                       estimador rPPG en vivo").
            amplification_factor: Color amplification factor for visual output.
            motion_threshold: Maximum landmark displacement (in normalized pixels,
                              i.e. already scaled by frame diagonal) before a frame
                              is rejected as motion-corrupted. MediaPipe's own
                              landmark jitter is ~2-5 raw px even when still, which
                              is why this must sit comfortably above that noise
                              floor rather than at raw-pixel scale (see
                              backend/app/references.py, HEART_RATE_MOTION_THRESHOLD).
            min_estimation_seconds: Required time span of buffered signal before
                              any BPM is reported. Measured from timestamps rather
                              than frame count so it holds at any real frame rate.
            min_confidence: Minimum spectral peak-to-band-power ratio for an
                              estimate to be accepted into the smoothing history.
            max_bpm_change_per_second: Physiological rate-of-change ceiling used to
                              reject implausible jumps between updates.
            stale_after_seconds: If no estimate clears min_confidence for this long,
                              report signal_ready=False instead of showing a stale
                              number.
            max_raw_spread_bpm: Maximum interquartile spread of recent unsmoothed
                              estimates for the reading to count as a real pulse.
                              A genuine pulse puts its peak at the same frequency
                              window after window; noise that happens to clear the
                              confidence gate lands somewhere different each time.
                              Without this check, heavy noise produced a rock-steady
                              but wrong reading (~105 BPM for a true 72), which
                              reads as more trustworthy than an obviously jumpy one.
            estimation_interval_seconds: Minimum wall-clock gap between spectral
                              estimates. Per-frame estimation re-ran the FFT 20
                              times a second over windows overlapping by ~95%,
                              so the resulting values were correlated by
                              construction and made the spread check above
                              uninformative. Spacing them out lets the history
                              span near-independent windows at a tenth of the cost.
        """
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.amplification_factor = amplification_factor
        self.buffer_size = int(buffer_seconds * fps)
        self.motion_threshold = motion_threshold
        self.min_estimation_seconds = min_estimation_seconds
        self.min_confidence = min_confidence
        self.max_bpm_change_per_second = max_bpm_change_per_second
        self.stale_after_seconds = stale_after_seconds
        self.max_raw_spread_bpm = max_raw_spread_bpm
        self.estimation_interval_seconds = estimation_interval_seconds

        # Signal buffers — multi-channel for CHROM
        self._red_signal: deque[float] = deque(maxlen=self.buffer_size)
        self._green_signal: deque[float] = deque(maxlen=self.buffer_size)
        self._blue_signal: deque[float] = deque(maxlen=self.buffer_size)
        self._timestamps: deque[float] = deque(maxlen=self.buffer_size)

        # Both histories are sized in estimates, which now arrive on a fixed
        # time interval rather than once per frame: 10 estimates of smoothing
        # (~5s) and 20 of spread (~10s, nearly the whole signal buffer, so the
        # oldest and newest come from windows that barely overlap).
        self._hr_history: deque[float] = deque(maxlen=10)

        # Unsmoothed accepted estimates, kept separately to measure how tightly
        # the spectral peak repeats. _hr_history cannot serve this purpose: the
        # rate-of-change limit deliberately removes the very spread being tested.
        self._raw_history: deque[float] = deque(maxlen=20)

        # Forehead ROI landmark indices (MediaPipe Face Mesh)
        # These form a polygon covering the forehead area
        self._forehead_indices = [
            10, 67, 69, 104, 108, 109, 151,
            337, 299, 333, 338, 297, 10
        ]

        # Nose-bridge reference ROI landmark indices
        # This region has minimal blood flow and captures common-mode noise
        self._nose_bridge_indices = [6, 197, 195, 5, 4]

        # Motion detection landmark indices (nose tip, forehead center)
        self._motion_landmark_indices = [1, 10]  # Nose tip, forehead center

        # State
        self._ready = False
        self._frame_count = 0
        self._last_bpm = 0.0
        self._signal_quality = 0.0
        self._warned_fs_low = False

        # Timestamp of the last estimate that cleared min_confidence, used to
        # decide whether the displayed BPM has gone stale, and of the last
        # accepted update, used for the rate-of-change limit.
        self._last_valid_ts: float | None = None
        self._last_update_ts: float | None = None

        # Spectral estimation is throttled, so the last computed values are
        # reused on the frames in between.
        self._last_estimate_ts: float | None = None
        self._last_signal_quality = 0.0

        # Motion artifact rejection state
        self._prev_motion_landmarks: list[tuple[float, float]] | None = None
        self._motion_frames_skipped = 0
        self._last_motion_detected = False

        # Slow-moving baseline of the nose-bridge reference ROI, used to
        # remove illumination drift without collapsing the forehead's
        # absolute brightness level (see _extract_nose_bridge_signal usage
        # in process_frame).
        self._ref_baseline: tuple[float, float, float] | None = None

    def process_frame(
        self,
        frame: np.ndarray,
        landmarks: list[dict],
        frame_shape: tuple[int, int],
        timestamp: float,
    ) -> dict:
        """
        Process a single frame and update the heart rate estimate.

        Args:
            frame: BGR frame from OpenCV.
            landmarks: MediaPipe face landmarks (normalized 0-1).
            frame_shape: (height, width) of the frame.
            timestamp: Current timestamp in seconds.

        Returns:
            {
                "bpm": float,               # Estimated heart rate in BPM
                "bpm_confidence": float,     # Signal quality (0-1)
                "signal_ready": bool,        # Whether enough data is collected
                "raw_signal_value": float,   # Current green channel mean
                "stress_indicator": float,   # 0-1 based on HR variability
                "motion_detected": bool,     # Whether motion artifact was detected
                "signal_quality": float,     # SNR-based signal reliability (0-1)
            }
        """
        if not SCIPY_AVAILABLE:
            return self._empty_result()

        self._frame_count += 1

        # Step 1: Motion artifact detection
        motion_detected = self._detect_motion(landmarks, frame_shape)
        self._last_motion_detected = motion_detected

        # Step 2: Extract forehead ROI (multi-channel)
        roi_rgb = self._extract_forehead_signal(frame, landmarks, frame_shape)
        if roi_rgb is None:
            return self._empty_result()

        r_mean, g_mean, b_mean = roi_rgb

        # Step 3: Extract nose-bridge reference and subtract common-mode noise
        ref_rgb = self._extract_nose_bridge_signal(frame, landmarks, frame_shape)
        if ref_rgb is not None:
            ref_r, ref_g, ref_b = ref_rgb
            if self._ref_baseline is None:
                self._ref_baseline = (ref_r, ref_g, ref_b)
            base_r, base_g, base_b = self._ref_baseline

            # Subtract only the *drift* of the reference ROI relative to its
            # own baseline — not its raw level. Subtracting the raw nose
            # value (as before) routinely made r/g/b_mean negative (nose is
            # usually brighter than forehead), which tripped the near-zero
            # guard in _compute_chrom_signal and made bpm stick at 0. Drift
            # correction removes the same common-mode illumination change
            # while keeping r/g/b_mean near the forehead's real ~0-255 level,
            # which CHROM's per-channel mean normalization assumes.
            r_mean = r_mean - (ref_r - base_r)
            g_mean = g_mean - (ref_g - base_g)
            b_mean = b_mean - (ref_b - base_b)

            # Track slow lighting drift with an exponential moving average
            # so genuine illumination changes are still cancelled over time.
            ema = 0.01
            self._ref_baseline = (
                (1 - ema) * base_r + ema * ref_r,
                (1 - ema) * base_g + ema * ref_g,
                (1 - ema) * base_b + ema * ref_b,
            )

        # Step 4: If motion detected, skip adding to buffer
        if motion_detected:
            self._motion_frames_skipped += 1
            # Still return current estimate but flag motion
            return self._build_result(
                bpm=round(self._last_bpm, 1),
                confidence=round(self._signal_quality, 3),
                signal_ready=self._ready,
                raw_signal_value=round(g_mean, 4),
                motion_detected=True,
            )

        # Step 5: Add to signal buffers
        self._red_signal.append(r_mean)
        self._green_signal.append(g_mean)
        self._blue_signal.append(b_mean)
        self._timestamps.append(timestamp)

        # Step 6: Require a long enough *time span* of signal, measured from the
        # timestamps rather than a frame count so the requirement holds at any
        # real frame rate. The previous 3-second threshold gave the FFT a
        # resolution of only 20 BPM per bin at 20 FPS, so a reading could land
        # on 60, 80, 100 ... and nothing in between.
        buffer_span = (
            self._timestamps[-1] - self._timestamps[0]
            if len(self._timestamps) > 1
            else 0.0
        )
        if buffer_span < self.min_estimation_seconds or len(self._green_signal) < 40:
            return {
                "bpm": 0.0,
                "bpm_confidence": 0.0,
                "signal_ready": False,
                "raw_signal_value": round(g_mean, 4),
                "stress_indicator": 0.0,
                "motion_detected": False,
                "signal_quality": 0.0,
            }

        # Step 7: Estimate heart rate using CHROM algorithm, on a fixed interval
        # rather than every frame. Between estimates the previous values stand.
        due = (
            self._last_estimate_ts is None
            or (timestamp - self._last_estimate_ts) >= self.estimation_interval_seconds
        )
        if due:
            self._last_estimate_ts = timestamp
            bpm, confidence = self._estimate_heart_rate()
            self._last_signal_quality = self._compute_signal_quality()

            # Step 8: Accept the estimate only when the spectral peak stands out
            # from the noise floor, then hold it to a physiological rate of
            # change. Previously any bpm > 0 entered the history regardless of
            # confidence, so a noise peak anywhere in the band was displayed as
            # a real reading.
            if bpm > 0 and confidence >= self.min_confidence:
                self._raw_history.append(bpm)
                self._hr_history.append(self._limit_rate_of_change(bpm, timestamp))
                self._last_valid_ts = timestamp
                self._last_update_ts = timestamp
        else:
            confidence = self._signal_quality

        smoothed_bpm = float(np.median(self._hr_history)) if self._hr_history else 0.0
        self._last_bpm = smoothed_bpm

        # A number is only shown while a recent confident estimate backs it.
        # Showing the last good value indefinitely would read as current to the
        # interviewer long after the signal was lost.
        is_fresh = (
            self._last_valid_ts is not None
            and (timestamp - self._last_valid_ts) <= self.stale_after_seconds
        )
        self._ready = bool(self._hr_history) and is_fresh and self._peak_is_consistent()

        # Step 9: Compute stress indicator from HR level
        stress = self._compute_stress_indicator()
        self._signal_quality = confidence

        return {
            "bpm": round(smoothed_bpm, 1),
            "bpm_confidence": round(confidence, 3),
            "signal_ready": self._ready,
            "raw_signal_value": round(g_mean, 4),
            "stress_indicator": round(stress, 3),
            "motion_detected": False,
            "signal_quality": round(self._last_signal_quality, 3),
        }

    def _peak_is_consistent(self) -> bool:
        """
        Whether recent unsmoothed estimates agree closely enough to be a pulse.

        The confidence gate alone cannot separate a weak pulse from structured
        noise: under heavy noise both score around 0.38-0.40. What does separate
        them is repetition — a real pulse lands on the same frequency window
        after window, so the interquartile spread of recent estimates stays
        small, while noise peaks scatter across the band.
        """
        if len(self._raw_history) < 8:
            return False

        raw = np.array(self._raw_history)
        spread = float(np.percentile(raw, 75) - np.percentile(raw, 25))
        return spread <= self.max_raw_spread_bpm

    def _limit_rate_of_change(self, bpm: float, timestamp: float) -> float:
        """
        Clamp a new estimate to a physiologically reachable change since the
        last accepted one.

        Heart rate cannot move 60 BPM in a second, so a jump that large is
        always an estimation artifact rather than a measurement. Clamping lets
        the estimate converge on a genuine shift over a second or two while
        refusing to follow single-frame spikes.
        """
        if self._last_bpm <= 0 or self._last_update_ts is None:
            return bpm

        elapsed = max(timestamp - self._last_update_ts, 1e-3)
        max_delta = self.max_bpm_change_per_second * elapsed
        return float(
            np.clip(bpm, self._last_bpm - max_delta, self._last_bpm + max_delta)
        )

    def get_magnified_frame(
        self,
        frame: np.ndarray,
        landmarks: list[dict],
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Return the frame with Eulerian color magnification applied to the
        forehead region, making the pulse visible.

        This is for visual feedback in the dashboard — the viewer can
        actually see the blood flow pulsing through the skin.

        The effect works by:
          1. Computing the CHROM pulse signal from R, G, B channel histories
          2. Bandpass filtering the combined CHROM signal
          3. Normalizing the filtered value to [-1, +1]
          4. Mapping pulse phase to a vivid red ↔ cyan color shift
          5. Alpha-blending the color overlay onto the forehead ROI
          6. Drawing a glowing contour around the forehead region
          7. Adding an on-screen BPM readout
        """
        if not SCIPY_AVAILABLE or len(self._green_signal) < int(self.fps * 3):
            return frame

        h, w = frame_shape

        # Get forehead polygon
        forehead_pts = self._get_forehead_polygon(landmarks, h, w)
        if forehead_pts is None:
            return frame

        # Create forehead mask with feathered (blurred) edges
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [forehead_pts], 255)
        mask_blurred = cv2.GaussianBlur(mask, (21, 21), 10)
        mask_float = mask_blurred.astype(np.float32) / 255.0

        # Compute the CHROM pulse signal and apply bandpass filter
        chrom_signal = self._compute_chrom_signal()
        if chrom_signal is None or len(chrom_signal) == 0:
            return frame

        # Detrend before filtering (remove DC / slow drift)
        chrom_signal = chrom_signal - np.mean(chrom_signal)
        filtered = self._bandpass_filter(chrom_signal)

        if filtered is None or len(filtered) == 0:
            return frame

        # ── Normalize the pulse signal to [-1, +1] ──────────────────
        # Use the recent standard deviation for consistent normalization
        std_val = np.std(filtered)
        if std_val < 1e-6:
            # No meaningful variation yet — return frame with ROI outline only
            output = frame.copy()
            cv2.polylines(output, [forehead_pts], True, (255, 255, 0), 1, cv2.LINE_AA)
            return output

        normalized_pulse = np.clip(filtered[-1] / (std_val * 2.0), -1.0, 1.0)

        # ── Map pulse phase to a vivid color ─────────────────────────
        # positive pulse (systole / blood inflow)  → warm red tint
        # negative pulse (diastole / blood outflow) → cool cyan tint
        # This makes the pulsing dramatically visible
        alpha_strength = 0.55  # Overlay opacity (0.0–1.0)

        if normalized_pulse > 0:
            # Red-ish overlay: BGR = (0, 0, 255) scaled by pulse strength
            color = np.array([0, 30, 255], dtype=np.float32) * abs(normalized_pulse)
        else:
            # Cyan-ish overlay: BGR = (200, 180, 0) scaled by pulse strength
            color = np.array([200, 180, 0], dtype=np.float32) * abs(normalized_pulse)

        # Build the 3-channel overlay
        overlay = frame.copy().astype(np.float32)
        for c in range(3):
            overlay[:, :, c] += color[c] * mask_float * alpha_strength

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # ── Draw pulsing contour around the forehead ROI ─────────────
        # Contour color pulses between cyan and red with the heartbeat
        pulse_lerp = (normalized_pulse + 1.0) / 2.0  # Map to [0, 1]
        contour_b = int(200 * (1.0 - pulse_lerp))
        contour_g = int(180 * (1.0 - pulse_lerp) + 50)
        contour_r = int(255 * pulse_lerp)
        contour_color = (contour_b, contour_g, contour_r)

        # Thicker line that pulses
        thickness = 2 + int(abs(normalized_pulse) * 2)
        cv2.polylines(overlay, [forehead_pts], True, contour_color, thickness, cv2.LINE_AA)

        # ── On-screen BPM readout ────────────────────────────────────
        if self._last_bpm > 0:
            bpm_text = f"HR: {int(self._last_bpm)} BPM"
            # Position above the forehead ROI
            text_x = int(np.mean(forehead_pts[:, 0])) - 50
            text_y = max(int(np.min(forehead_pts[:, 1])) - 15, 20)

            # Background rectangle for readability
            (tw, th), _ = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (text_x - 4, text_y - th - 6),
                          (text_x + tw + 4, text_y + 4),
                          (0, 0, 0), -1)
            cv2.putText(overlay, bpm_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, contour_color, 2, cv2.LINE_AA)

        return overlay

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ══════════════════════════════════════════════════════════════════

    def _extract_forehead_signal(
        self,
        frame: np.ndarray,
        landmarks: list[dict],
        frame_shape: tuple[int, int],
    ) -> tuple[float, float, float] | None:
        """
        Extract the mean R, G, B channel values from the forehead ROI.
        
        All three channels are needed for the CHROM algorithm, which
        combines them via chrominance analysis to isolate the pulse signal
        more robustly than green-channel-only approaches.
        """
        h, w = frame_shape

        forehead_pts = self._get_forehead_polygon(landmarks, h, w)
        if forehead_pts is None:
            return None

        # Create mask for forehead
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [forehead_pts], 255)

        # Extract R, G, B channel means within the ROI (frame is BGR)
        blue_channel = frame[:, :, 0]
        green_channel = frame[:, :, 1]
        red_channel = frame[:, :, 2]

        roi_mask = mask == 255
        roi_blue = blue_channel[roi_mask]
        roi_green = green_channel[roi_mask]
        roi_red = red_channel[roi_mask]

        if len(roi_green) == 0:
            return None

        return (
            float(np.mean(roi_red)),
            float(np.mean(roi_green)),
            float(np.mean(roi_blue)),
        )

    def _extract_nose_bridge_signal(
        self,
        frame: np.ndarray,
        landmarks: list[dict],
        frame_shape: tuple[int, int],
    ) -> tuple[float, float, float] | None:
        """
        Extract mean R, G, B values from the nose-bridge reference ROI.
        
        The nose bridge has minimal blood flow pulsation and is used as
        a reference region. Subtracting this signal from the forehead
        signal removes common-mode noise such as lighting changes and
        camera auto-exposure adjustments.
        """
        h, w = frame_shape

        if not landmarks or len(landmarks) < 200:
            return None

        # Build nose-bridge polygon
        points = []
        for idx in self._nose_bridge_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px = int(lm["x"] * w)
                py = int(lm["y"] * h)
                points.append([px, py])

        if len(points) < 3:
            return None

        nose_pts = np.array(points, dtype=np.int32)

        # Create mask for nose bridge
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [nose_pts], 255)

        # Extract R, G, B channel means within the ROI (frame is BGR)
        roi_mask = mask == 255
        roi_blue = frame[:, :, 0][roi_mask]
        roi_green = frame[:, :, 1][roi_mask]
        roi_red = frame[:, :, 2][roi_mask]

        if len(roi_green) == 0:
            return None

        return (
            float(np.mean(roi_red)),
            float(np.mean(roi_green)),
            float(np.mean(roi_blue)),
        )

    def _detect_motion(
        self,
        landmarks: list[dict],
        frame_shape: tuple[int, int],
    ) -> bool:
        """
        Detect motion artifacts using landmark velocity between frames.
        
        Computes the mean displacement of key landmarks (nose tip, forehead
        center) between the current and previous frames. If displacement
        exceeds the motion threshold, the frame is marked as motion-corrupted
        and should not be added to the signal buffer.
        
        Returns True if motion artifact is detected.
        """
        h, w = frame_shape

        if not landmarks or len(landmarks) < max(self._motion_landmark_indices) + 1:
            self._prev_motion_landmarks = None
            return False

        # Extract current positions of motion landmarks
        current_positions = []
        for idx in self._motion_landmark_indices:
            lm = landmarks[idx]
            # Use normalized coordinates scaled by frame dimensions
            px = lm["x"] * w
            py = lm["y"] * h
            current_positions.append((px, py))

        if self._prev_motion_landmarks is None:
            self._prev_motion_landmarks = current_positions
            return False

        # Compute mean displacement
        total_displacement = 0.0
        for (cx, cy), (px, py) in zip(current_positions, self._prev_motion_landmarks):
            dx = cx - px
            dy = cy - py
            total_displacement += np.sqrt(dx * dx + dy * dy)

        mean_displacement = total_displacement / len(current_positions)

        # Normalize displacement by frame diagonal for resolution independence
        frame_diagonal = np.sqrt(w * w + h * h)
        normalized_displacement = mean_displacement / frame_diagonal * 1000.0

        # Update previous landmarks
        self._prev_motion_landmarks = current_positions

        # Check against threshold
        return normalized_displacement > self.motion_threshold

    def _get_forehead_polygon(
        self,
        landmarks: list[dict],
        h: int,
        w: int,
    ) -> np.ndarray | None:
        """Get the forehead polygon in pixel coordinates."""
        if not landmarks or len(landmarks) < 340:
            return None

        points = []
        for idx in self._forehead_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px = int(lm["x"] * w)
                py = int(lm["y"] * h)
                points.append([px, py])

        if len(points) < 3:
            return None

        return np.array(points, dtype=np.int32)

    def _compute_chrom_signal(self) -> np.ndarray | None:
        """
        Compute the CHROM (Chrominance-based) pulse signal from R, G, B buffers.
        
        Implements the method by de Haan & Jeanne (2013):
        - Normalize each channel by its mean to remove DC offset
        - Combine via chrominance projection:
            X = 3*R_n - 2*G_n
            Y = 1.5*R_n + G_n - 1.5*B_n
            alpha = std(X) / std(Y)
            pulse = X - alpha * Y
        
        This approach is more robust to motion and illumination changes
        than single-channel (green-only) methods.
        """
        if len(self._red_signal) < 10:
            return None

        R = np.array(self._red_signal)
        G = np.array(self._green_signal)
        B = np.array(self._blue_signal)

        # Normalize each channel by its mean
        r_mean = np.mean(R)
        g_mean = np.mean(G)
        b_mean = np.mean(B)

        if r_mean < 1e-6 or g_mean < 1e-6 or b_mean < 1e-6:
            return None

        R_n = R / r_mean
        G_n = G / g_mean
        B_n = B / b_mean

        # CHROM combination
        X = 3.0 * R_n - 2.0 * G_n
        Y = 1.5 * R_n + G_n - 1.5 * B_n

        std_x = np.std(X)
        std_y = np.std(Y)

        alpha = std_x / (std_y + 1e-6)

        pulse_signal = X - alpha * Y

        return pulse_signal

    def _estimate_heart_rate(self) -> tuple[float, float]:
        """
        Estimate heart rate using FFT on the CHROM pulse signal
        after bandpass filtering.
        
        Returns (bpm, confidence).
        """
        # Compute CHROM pulse signal from multi-channel data
        chrom_signal = self._compute_chrom_signal()
        if chrom_signal is None:
            return 0.0, 0.0

        # Remove DC and linear drift. The Butterworth bandpass is deliberately
        # not applied here: the band mask below already isolates the frequencies
        # of interest, while the filter's own response peaks inside that band,
        # which biased noise-only spectra toward a repeatable ~105 BPM peak that
        # the consistency check could not tell apart from a real pulse. The
        # filter is still used for the visual magnification path.
        filtered = detrend(chrom_signal, type="linear")

        # Compute FFT
        n = len(filtered)

        # Estimate actual FPS from timestamps
        if len(self._timestamps) > 1:
            time_span = self._timestamps[-1] - self._timestamps[0]
            actual_fps = (len(self._timestamps) - 1) / max(time_span, 0.1)
        else:
            actual_fps = self.fps

        # Hann window before the transform. A rectangular window's abrupt edges
        # leak the pulse peak's energy into neighbouring bins, which is part of
        # why the argmax hopped between adjacent frequencies frame to frame.
        windowed = filtered * np.hanning(n)

        # Zero-pad to evaluate the spectrum on a 4x finer grid. This adds no
        # information, but it removes the bin quantization that otherwise forces
        # the reported BPM onto multiples of 60*fs/n (20 BPM at 3s/20FPS).
        n_fft = n * 4

        freqs = fftfreq(n_fft, d=1.0 / actual_fps)
        fft_power = np.abs(fft(windowed, n=n_fft)) ** 2

        # Only look at positive frequencies in our band of interest
        valid_mask = (freqs > self.freq_low) & (freqs < self.freq_high)
        valid_freqs = freqs[valid_mask]
        valid_power = fft_power[valid_mask]

        if len(valid_power) == 0:
            return 0.0, 0.0

        # Find dominant frequency
        peak_idx = int(np.argmax(valid_power))
        peak_freq = valid_freqs[peak_idx]

        # Convert frequency to BPM
        bpm = peak_freq * 60.0

        # Confidence = fraction of in-band power concentrated around the peak.
        # Defined over a fixed frequency neighbourhood (in Hz) rather than a bin
        # count so it stays comparable regardless of window length or padding;
        # a peak-to-total-bin ratio would shrink fourfold purely from the
        # zero-padding above. Pure tone -> near 1.0, white noise -> ~0.2.
        total_power = float(np.sum(valid_power))
        if total_power <= 0:
            return 0.0, 0.0

        peak_band = np.abs(valid_freqs - peak_freq) <= 0.2
        confidence = float(np.sum(valid_power[peak_band]) / total_power)

        # The band itself is now restricted to physiologically reachable rates,
        # so a peak inside it needs no additional range penalty.
        return float(bpm), float(np.clip(confidence, 0.0, 1.0))

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray | None:
        """Apply Butterworth bandpass filter to isolate pulse frequency."""
        if len(signal) < 10:
            return None

        # Estimate FPS from timestamps
        if len(self._timestamps) > 1:
            time_span = self._timestamps[-1] - self._timestamps[0]
            fs = (len(self._timestamps) - 1) / max(time_span, 0.1)
        else:
            fs = self.fps

        nyquist = fs / 2.0

        # Dynamically adjust cutoff frequencies to fit under Nyquist frequency
        low = self.freq_low / nyquist
        high = self.freq_high / nyquist

        # If low frequency exceeds Nyquist limit, clamp it to a safe low ratio
        if low >= 1.0 or low <= 0:
            low = 0.05  # Fallback to extremely low frequency (3 BPM)
            
        # If high frequency exceeds Nyquist limit, clamp it to 90% of Nyquist to avoid instability
        if high >= 1.0 or high <= 0:
            high = 0.90
            if not self._warned_fs_low:
                print(f"[WARN] Sampling rate fs={fs:.2f} FPS is too low for high cutoff frequency {self.freq_high} Hz. Clamped to {high * nyquist:.2f} Hz.")
                self._warned_fs_low = True

        # Ensure low is strictly less than high
        if low >= high:
            low = 0.1
            high = 0.9

        try:
            b, a = butter(3, [low, high], btype='band')
            filtered = filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))
            return filtered
        except Exception as e:
            print(f"[ERR] Butterworth filter failed: {e}")
            return None

    def _compute_stress_indicator(self) -> float:
        """
        Compute a stress indicator from the heart rate level.

        Only the HR-level term survives here. The previous version also scored
        the standard deviation of this same BPM history as if it were heart rate
        variability, which it never was: HRV requires beat-to-beat R-R
        intervals, whereas this history holds spectral estimates from
        overlapping windows. Its spread measures estimator noise, so the rule
        "low spread = stress" actually rewarded a poor camera signal — which
        produces wild estimates — with a "relaxed" reading. Smoothing the BPM
        output removed that spread almost entirely, which would have pinned the
        term near its maximum and reported constant stress instead.

        The direction that remains (elevated HR accompanies sympathetic
        activation) is the part with support; the cutoffs are still heuristic
        (see references.py, "Mezcla de estrés fisiológico").
        """
        if len(self._hr_history) < 3:
            return 0.0

        mean_hr = float(np.mean(self._hr_history))

        # 70 = calm baseline, 120 = clearly elevated
        return float(np.clip((mean_hr - 70) / 50.0, 0.0, 1.0))

    def _compute_signal_quality(self) -> float:
        """
        Compute signal quality using the Signal-to-Noise Ratio (SNR) of
        the power spectral density of the filtered CHROM pulse signal.
        
        The SNR is calculated as the ratio of the peak power (at the
        dominant pulse frequency) to the average noise floor power
        across the valid frequency band.
        
        Returns:
            A value between 0.0 and 1.0 representing signal reliability:
            - 0.0 = very noisy, unreliable signal
            - 1.0 = clean, high-confidence signal
        """
        chrom_signal = self._compute_chrom_signal()
        if chrom_signal is None or len(chrom_signal) < 10:
            return 0.0

        # Detrend
        signal = chrom_signal - np.mean(chrom_signal)

        # Apply bandpass filter
        filtered = self._bandpass_filter(signal)
        if filtered is None:
            return 0.0

        n = len(filtered)

        # Estimate actual FPS
        if len(self._timestamps) > 1:
            time_span = self._timestamps[-1] - self._timestamps[0]
            actual_fps = (len(self._timestamps) - 1) / max(time_span, 0.1)
        else:
            actual_fps = self.fps

        # Compute power spectral density via FFT
        freqs = fftfreq(n, d=1.0 / actual_fps)
        fft_values = np.abs(fft(filtered)) ** 2  # Power spectrum

        # Focus on valid frequency band
        valid_mask = (freqs > self.freq_low) & (freqs < self.freq_high)
        valid_psd = fft_values[valid_mask]

        if len(valid_psd) == 0:
            return 0.0

        # Peak power
        peak_power = np.max(valid_psd)

        # Average noise floor (mean of all bins excluding the peak)
        if len(valid_psd) > 1:
            # Remove the peak bin to estimate noise floor
            noise_bins = np.delete(valid_psd, np.argmax(valid_psd))
            noise_floor = np.mean(noise_bins)
        else:
            noise_floor = valid_psd[0]

        if noise_floor < 1e-10:
            return 1.0  # No noise detected, perfect signal (unlikely)

        # SNR in linear scale
        snr = peak_power / noise_floor

        # Map SNR to 0.0–1.0 quality scale
        # SNR of 1 = noise (quality 0), SNR of 10+ = good (quality ~1)
        quality = float(np.clip((snr - 1.0) / 9.0, 0.0, 1.0))

        return quality

    def _build_result(
        self,
        bpm: float,
        confidence: float,
        signal_ready: bool,
        raw_signal_value: float,
        motion_detected: bool,
    ) -> dict:
        """Build a result dict with current stress and signal quality."""
        stress = self._compute_stress_indicator() if signal_ready else 0.0
        signal_quality = self._last_signal_quality if signal_ready else 0.0
        return {
            "bpm": bpm,
            "bpm_confidence": confidence,
            "signal_ready": signal_ready,
            "raw_signal_value": raw_signal_value,
            "stress_indicator": round(stress, 3),
            "motion_detected": motion_detected,
            "signal_quality": round(signal_quality, 3),
        }

    def _empty_result(self) -> dict:
        """Return empty result when analysis isn't possible."""
        return {
            "bpm": 0.0,
            "bpm_confidence": 0.0,
            "signal_ready": False,
            "raw_signal_value": 0.0,
            "stress_indicator": 0.0,
            "motion_detected": False,
            "signal_quality": 0.0,
        }
