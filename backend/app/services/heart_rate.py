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
    from scipy.signal import butter, filtfilt
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
        buffer_seconds: float = 10.0,
        fps: float = 15.0,
        freq_low: float = 0.7,    # 42 BPM minimum
        freq_high: float = 4.0,   # 240 BPM maximum
        amplification_factor: float = 50.0,
        motion_threshold: float = 5.0,  # Normalized pixel displacement threshold
    ):
        """
        Args:
            buffer_seconds: How many seconds of signal to keep for analysis.
            fps: Expected frames per second (for frequency calculations).
            freq_low: Low cutoff frequency in Hz (0.7 = 42 BPM).
            freq_high: High cutoff frequency in Hz (4.0 = 240 BPM).
            amplification_factor: Color amplification factor for visual output.
            motion_threshold: Maximum landmark displacement (in normalized pixels)
                              before a frame is rejected as motion-corrupted.
        """
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.amplification_factor = amplification_factor
        self.buffer_size = int(buffer_seconds * fps)
        self.motion_threshold = motion_threshold

        # Signal buffers — multi-channel for CHROM
        self._red_signal: deque[float] = deque(maxlen=self.buffer_size)
        self._green_signal: deque[float] = deque(maxlen=self.buffer_size)
        self._blue_signal: deque[float] = deque(maxlen=self.buffer_size)
        self._timestamps: deque[float] = deque(maxlen=self.buffer_size)

        # Heart rate history for smoothing
        self._hr_history: deque[float] = deque(maxlen=10)

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

        # Motion artifact rejection state
        self._prev_motion_landmarks: list[tuple[float, float]] | None = None
        self._motion_frames_skipped = 0
        self._last_motion_detected = False

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
            # Subtract reference signal to remove common-mode noise
            # (lighting changes, camera auto-exposure)
            r_mean = r_mean - ref_r
            g_mean = g_mean - ref_g
            b_mean = b_mean - ref_b

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

        # Step 6: Need minimum data to estimate HR
        min_samples = int(self.fps * 3)  # At least 3 seconds
        if len(self._green_signal) < min_samples:
            return {
                "bpm": 0.0,
                "bpm_confidence": 0.0,
                "signal_ready": False,
                "raw_signal_value": round(g_mean, 4),
                "stress_indicator": 0.0,
                "motion_detected": False,
                "signal_quality": 0.0,
            }

        self._ready = True

        # Step 7: Estimate heart rate using CHROM algorithm
        bpm, confidence = self._estimate_heart_rate()

        # Step 8: Smooth the BPM output
        if bpm > 0:
            self._hr_history.append(bpm)

        smoothed_bpm = float(np.median(self._hr_history)) if self._hr_history else 0.0
        self._last_bpm = smoothed_bpm

        # Step 9: Compute stress indicator from HR variability
        stress = self._compute_stress_indicator()

        # Step 10: Compute signal quality (SNR)
        signal_quality = self._compute_signal_quality()
        self._signal_quality = confidence

        return {
            "bpm": round(smoothed_bpm, 1),
            "bpm_confidence": round(confidence, 3),
            "signal_ready": True,
            "raw_signal_value": round(g_mean, 4),
            "stress_indicator": round(stress, 3),
            "motion_detected": False,
            "signal_quality": round(signal_quality, 3),
        }

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

        # Detrend the signal (remove DC component and slow drift)
        signal = chrom_signal - np.mean(chrom_signal)

        # Apply bandpass filter
        filtered = self._bandpass_filter(signal)
        if filtered is None:
            return 0.0, 0.0

        # Compute FFT
        n = len(filtered)
        
        # Estimate actual FPS from timestamps
        if len(self._timestamps) > 1:
            time_span = self._timestamps[-1] - self._timestamps[0]
            actual_fps = (len(self._timestamps) - 1) / max(time_span, 0.1)
        else:
            actual_fps = self.fps

        freqs = fftfreq(n, d=1.0 / actual_fps)
        fft_values = np.abs(fft(filtered))

        # Only look at positive frequencies in our band of interest
        valid_mask = (freqs > self.freq_low) & (freqs < self.freq_high)
        valid_freqs = freqs[valid_mask]
        valid_fft = fft_values[valid_mask]

        if len(valid_fft) == 0:
            return 0.0, 0.0

        # Find dominant frequency
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_fft[peak_idx]

        # Convert frequency to BPM
        bpm = peak_freq * 60.0

        # Confidence = ratio of peak power to total power (signal-to-noise)
        total_power = np.sum(valid_fft)
        confidence = float(peak_power / total_power) if total_power > 0 else 0.0

        # Sanity check BPM range
        if bpm < 40 or bpm > 200:
            confidence *= 0.3  # Low confidence for extreme values

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
        Compute a stress indicator based on heart rate variability (HRV).
        
        Physiological basis (corrected):
        - **High HRV** (high std of R-R intervals) indicates good vagal tone
          and parasympathetic activity → RELAXATION.
        - **Low HRV** (low std) indicates sympathetic dominance → STRESS.
        
        This is consistent with established cardiology research: reduced HRV
        is a marker of stress, anxiety, and poor cardiovascular health.
        
        Combined factors:
        - Factor 1: Elevated heart rate (higher HR = more stress)
        - Factor 2: Low variability = stress (INVERTED from naive assumption)
        """
        if len(self._hr_history) < 3:
            return 0.0

        hr_values = np.array(self._hr_history)
        mean_hr = np.mean(hr_values)
        std_hr = np.std(hr_values)

        # Factor 1: Elevated heart rate
        # 60-80 = calm (0.0), 80-100 = moderate (0.3-0.5), >100 = stressed (0.7-1.0)
        hr_stress = float(np.clip((mean_hr - 70) / 50.0, 0.0, 1.0))

        # Factor 2: LOW variability = stress (physiologically correct)
        # High std (>8) = relaxed (0.0), Low std (<2) = stressed (1.0)
        variability_stress = float(np.clip(1.0 - (std_hr / 10.0), 0.0, 1.0))

        # Combined stress indicator
        stress = hr_stress * 0.6 + variability_stress * 0.4
        return float(np.clip(stress, 0.0, 1.0))

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
        signal_quality = self._compute_signal_quality() if signal_ready else 0.0
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
