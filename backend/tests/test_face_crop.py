import cv2
import numpy as np

from backend.app.services.emotion_classifier import EmotionClassifier


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(360, 480, 3), dtype=np.uint8)


BBOX = {"x_min": 100, "y_min": 50, "x_max": 300, "y_max": 290}


def _old_preprocess(frame_bgr, bbox):
    """Verbatim copy of preprocess_face before the refactor."""
    x1 = max(0, bbox.get("x_min", 0))
    y1 = max(0, bbox.get("y_min", 0))
    x2 = bbox.get("x_max", 0)
    y2 = bbox.get("y_max", 0)
    inset_x = int((x2 - x1) * 0.10)
    inset_y = int((y2 - y1) * 0.10)
    x1, y1 = x1 + inset_x, y1 + inset_y
    x2, y2 = x2 - inset_x, y2 - inset_y
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    three = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    chw = three.transpose(2, 0, 1)
    chw = (chw - EmotionClassifier._CNN_MEAN) / EmotionClassifier._CNN_STD
    return chw[np.newaxis, ...].astype(np.float32)


def test_crop_face_gray_shape_and_dtype():
    gray = EmotionClassifier.crop_face_gray(_frame(), BBOX)
    assert gray.shape == (224, 224)
    assert gray.dtype == np.uint8


def test_crop_face_gray_empty_box_returns_none():
    assert EmotionClassifier.crop_face_gray(_frame(), {"x_min": 10, "y_min": 10, "x_max": 10, "y_max": 10}) is None


def test_preprocess_face_matches_previous_implementation():
    frame = _frame()
    np.testing.assert_array_equal(
        EmotionClassifier.preprocess_face(frame, BBOX), _old_preprocess(frame, BBOX))
