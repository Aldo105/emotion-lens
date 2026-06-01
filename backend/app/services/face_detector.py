"""
EmotionLens — Face Detector Service

Wraps MediaPipe FaceLandmarker (Tasks API, v0.10.35+) to provide:
- 478 3D facial landmarks (normalized and pixel coordinates)
- Face bounding box
- Key landmark groups for downstream analysis (eyes, brows, mouth, etc.)
- Face blendshapes (52 FACS-like coefficients) when available

Migrated from legacy mp.solutions.face_mesh to mp.tasks.vision.FaceLandmarker.
"""

import os
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[!!] MediaPipe not installed. Face detection will not work.")


# ── MediaPipe Landmark Index Groups ──────────────────────────────────
# These indices map to specific facial regions in MediaPipe's 478-point mesh.

LANDMARK_GROUPS = {
    # Eyebrows
    "left_inner_brow": 107,
    "right_inner_brow": 336,
    "left_outer_brow": 70,
    "right_outer_brow": 300,
    "left_mid_brow": 105,
    "right_mid_brow": 334,

    # Eyes
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "right_eye_top": 386,
    "right_eye_bottom": 374,

    # Nose
    "nose_tip": 1,
    "nose_bridge": 6,
    "left_nose_wing": 49,
    "right_nose_wing": 279,

    # Mouth / Lips
    "upper_lip_center": 13,
    "lower_lip_center": 14,
    "left_lip_corner": 61,
    "right_lip_corner": 291,
    "upper_lip_top": 0,
    "lower_lip_bottom": 17,
    "left_upper_lip": 40,
    "right_upper_lip": 270,

    # Jaw & Chin
    "chin": 152,
    "left_jaw": 58,
    "right_jaw": 288,
    "left_cheek": 50,
    "right_cheek": 280,

    # Forehead reference points
    "forehead_center": 10,
    "left_forehead": 67,
    "right_forehead": 297,
}

# Iris landmarks (indices 468-477 in the 478-point model)
IRIS_LANDMARKS = {
    "left_iris_center": 468,
    "right_iris_center": 473,
}


def _find_model_path() -> str:
    """Locate the face_landmarker.task model file."""
    # Check common locations relative to project root
    candidates = [
        os.path.join(os.getcwd(), "data", "models", "face_landmarker.task"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "models", "face_landmarker.task"),
    ]
    for path in candidates:
        resolved = os.path.abspath(path)
        if os.path.isfile(resolved):
            return resolved

    raise FileNotFoundError(
        "face_landmarker.task model not found. Please download it:\n"
        "  python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task', "
        "'data/models/face_landmarker.task')\""
    )


class FaceDetector:
    """
    MediaPipe FaceLandmarker wrapper for real-time face detection
    and landmark extraction.

    Uses the new mp.tasks API (MediaPipe >= 0.10.14).
    """

    def __init__(
        self,
        max_faces: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        model_path: str | None = None,
    ):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe is required but not installed.")

        if model_path is None:
            model_path = _find_model_path()

        print(f"[+] Loading FaceLandmarker model from: {model_path}")

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.refine_landmarks = refine_landmarks

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Detect face and extract landmarks from a BGR frame.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            Dict with landmarks and metadata, or None if no face found.
            {
                "landmarks": [{"x": float, "y": float, "z": float}, ...],  # 478 points (normalized)
                "landmarks_px": [{"x": int, "y": int, "z": float}, ...],   # pixel coords
                "bbox": {"x_min": int, "y_min": int, "x_max": int, "y_max": int},
                "key_points": {name: {"x": float, "y": float, "z": float}},
                "blendshapes": {name: float} or None,  # 52 FACS-like coefficients
                "frame_shape": (height, width),
            }
        """
        import cv2

        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        # MediaPipe Tasks expects RGB in its own Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        # Use the first (primary) face
        face_lms = results.face_landmarks[0]

        # Extract all landmarks
        landmarks = []
        landmarks_px = []
        xs, ys = [], []

        for lm in face_lms:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
            px_x = int(lm.x * w)
            px_y = int(lm.y * h)
            landmarks_px.append({"x": px_x, "y": px_y, "z": lm.z})
            xs.append(px_x)
            ys.append(px_y)

        # Bounding box
        bbox = {
            "x_min": max(0, min(xs)),
            "y_min": max(0, min(ys)),
            "x_max": min(w, max(xs)),
            "y_max": min(h, max(ys)),
        }

        # Extract key named landmarks
        key_points = {}
        for name, idx in LANDMARK_GROUPS.items():
            if idx < len(landmarks):
                key_points[name] = landmarks[idx]

        # Add iris landmarks
        if self.refine_landmarks:
            for name, idx in IRIS_LANDMARKS.items():
                if idx < len(landmarks):
                    key_points[name] = landmarks[idx]

        # Extract blendshapes if available
        blendshapes = None
        if results.face_blendshapes and len(results.face_blendshapes) > 0:
            blendshapes = {}
            for bs in results.face_blendshapes[0]:
                blendshapes[bs.category_name] = bs.score

        return {
            "landmarks": landmarks,
            "landmarks_px": landmarks_px,
            "bbox": bbox,
            "key_points": key_points,
            "blendshapes": blendshapes,
            "frame_shape": (h, w),
        }

    def get_landmark_array(self, detection_result: dict) -> np.ndarray:
        """
        Convert landmarks to a numpy array of shape (N, 3) for vectorized math.
        """
        lms = detection_result["landmarks"]
        return np.array([[lm["x"], lm["y"], lm["z"]] for lm in lms], dtype=np.float32)

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
