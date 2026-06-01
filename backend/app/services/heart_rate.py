"""
EmotionLens — Eulerian Video Magnification & Heart Rate Estimator

Integrates Eulerian Color Magnification to detect subtle skin color changes
caused by blood flow, enabling remote heart rate (rPPG) estimation.

Algorithm Pipeline:
  1. Extract forehead ROI using MediaPipe landmarks
  2. Average the green channel intensity across the ROI per frame
  3. Build a signal buffer (rolling window of ~10 seconds)
  4. Apply Butterworth bandpass filter (0.7 - 4.0 Hz = 42-240 BPM)
  5. FFT to find dominant frequency -> convert to BPM
  6. Optionally: amplify color changes for visual feedback

Based on MIT CSAIL's "Eulerian Video Magnification for Revealing
Subtle Changes in the World" (Wu et al., 2012).

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
    MediaPipe landmarks. Applies Eulerian color magnification principles:
    temporal bandpass filtering on the green channel to isolate pulse signal.
    """

    def __init__(
        self,
        buffer_seconds: float = 10.0,
        fps: float = 15.0,
        freq_low: float = 0.7,    # 42 BPM minimum
        freq_high: float = 4.0,   # 240 BPM maximum
        amplification_factor: float = 50.0,
    ):
        """
        Args:
            buffer_seconds: How many seconds of signal to keep for analysis.
            fps: Expected frames per second (for frequency calculations).
            freq_low: Low cutoff frequency in Hz (0.7 = 42 BPM).
            freq_high: High cutoff frequency in Hz (4.0 = 240 BPM).
            amplification_factor: Color amplification factor for visual output.
        """
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.amplification_factor = amplification_factor
        self.buffer_size = int(buffer_seconds * fps)

        # Signal buffers
        self._green_signal: deque[float] = deque(maxlen=self.buffer_size)
        self._timestamps: deque[float] = deque(maxlen=self.buffer_size)

        # Heart rate history for smoothing
        self._hr_history: deque[float] = deque(maxlen=10)

        # Forehead ROI landmark indices (MediaPipe Face Mesh)
        # These form a polygon covering the forehead area
        self._forehead_indices = [
            10, 67, 69, 104, 108, 109, 151,
            337, 299, 333, 338, 297, 10
        ]

        # State
        self._ready = False
        self._frame_count = 0
        self._last_bpm = 0.0
        self._signal_quality = 0.0
        self._warned_fs_low = False

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
            }
        """
        if not SCIPY_AVAILABLE:
            return self._empty_result()

        self._frame_count += 1

        # Step 1: Extract forehead ROI
        roi_mean = self._extract_forehead_signal(frame, landmarks, frame_shape)
        if roi_mean is None:
            return self._empty_result()

        # Step 2: Add to signal buffer
        self._green_signal.append(roi_mean)
        self._timestamps.append(timestamp)

        # Step 3: Need minimum data to estimate HR
        min_samples = int(self.fps * 3)  # At least 3 seconds
        if len(self._green_signal) < min_samples:
            return {
                "bpm": 0.0,
                "bpm_confidence": 0.0,
                "signal_ready": False,
                "raw_signal_value": roi_mean,
                "stress_indicator": 0.0,
            }

        self._ready = True

        # Step 4: Estimate heart rate
        bpm, confidence = self._estimate_heart_rate()

        # Step 5: Smooth the BPM output
        if bpm > 0:
            self._hr_history.append(bpm)

        smoothed_bpm = float(np.median(self._hr_history)) if self._hr_history else 0.0
        self._last_bpm = smoothed_bpm

        # Step 6: Compute stress indicator from HR variability
        stress = self._compute_stress_indicator()

        return {
            "bpm": round(smoothed_bpm, 1),
            "bpm_confidence": round(confidence, 3),
            "signal_ready": True,
            "raw_signal_value": round(roi_mean, 4),
            "stress_indicator": round(stress, 3),
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
          1. Bandpass filtering the green channel signal history
          2. Normalizing the filtered value to [-1, +1]
          3. Mapping pulse phase to a vivid red ↔ cyan color shift
          4. Alpha-blending the color overlay onto the forehead ROI
          5. Drawing a glowing contour around the forehead region
          6. Adding an on-screen BPM readout
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

        # Apply temporal bandpass filter to the green signal
        signal = np.array(self._green_signal)
        # Detrend before filtering (remove DC / slow drift)
        signal = signal - np.mean(signal)
        filtered = self._bandpass_filter(signal)

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
    ) -> float | None:
        """
        Extract the mean green channel value from the forehead ROI.
        
        The green channel is most sensitive to hemoglobin absorption
        changes caused by blood flow.
        """
        h, w = frame_shape

        forehead_pts = self._get_forehead_polygon(landmarks, h, w)
        if forehead_pts is None:
            return None

        # Create mask for forehead
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [forehead_pts], 255)

        # Extract green channel mean within the ROI
        green_channel = frame[:, :, 1]  # BGR -> G is index 1
        roi_pixels = green_channel[mask == 255]

        if len(roi_pixels) == 0:
            return None

        return float(np.mean(roi_pixels))

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

    def _estimate_heart_rate(self) -> tuple[float, float]:
        """
        Estimate heart rate using FFT on the bandpass-filtered signal.
        
        Returns (bpm, confidence).
        """
        signal = np.array(self._green_signal)

        # Detrend the signal (remove DC component and slow drift)
        signal = signal - np.mean(signal)

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
        
        Higher HRV variation = more stress.
        Normal resting HR: 60-100 BPM.
        Elevated HR + high variability = stress signal.
        """
        if len(self._hr_history) < 3:
            return 0.0

        hr_values = np.array(self._hr_history)
        mean_hr = np.mean(hr_values)
        std_hr = np.std(hr_values)

        # Factor 1: Elevated heart rate
        # 60-80 = calm (0.0), 80-100 = moderate (0.3-0.5), >100 = stressed (0.7-1.0)
        hr_stress = float(np.clip((mean_hr - 70) / 50.0, 0.0, 1.0))

        # Factor 2: HR variability (high std = more stress/anxiety)
        # Low std (< 3) = calm, high std (> 10) = stressed
        variability_stress = float(np.clip(std_hr / 15.0, 0.0, 1.0))

        # Combined stress indicator
        stress = hr_stress * 0.6 + variability_stress * 0.4
        return float(np.clip(stress, 0.0, 1.0))

    def _empty_result(self) -> dict:
        """Return empty result when analysis isn't possible."""
        return {
            "bpm": 0.0,
            "bpm_confidence": 0.0,
            "signal_ready": False,
            "raw_signal_value": 0.0,
            "stress_indicator": 0.0,
        }
