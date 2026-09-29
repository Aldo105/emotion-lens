"""
The page's neutral calibration reaches the CNN: each person's resting-face
probability profile is subtracted from what the model reports afterwards.

No real model is loaded. The classifier is put in "cnn" mode and _predict_cnn
is replaced by a fake that plays back prepared outputs, so these tests pin the
calibration arithmetic and wiring, not the network.
"""

import numpy as np
import pytest

from backend.app.services.emotion_classifier import EmotionClassifier, FER7_LABELS

FACE = np.zeros((1, 3, 224, 224), np.float32)


def _probs(**given) -> dict[str, float]:
    """A 7-class distribution: the named labels, the remainder spread evenly."""
    rest = [k for k in FER7_LABELS if k not in given]
    share = (1.0 - sum(given.values())) / len(rest)
    return {**{k: share for k in rest}, **given}


def _cnn_output(probs: dict[str, float]) -> dict:
    top = max(probs, key=probs.get)
    return {
        "emotion": top,
        "confidence": probs[top],
        "probabilities": probs,
        "model_confidence": 0.5,
        "mode": "cnn",
    }


def _classifier(outputs: list[dict[str, float]]) -> EmotionClassifier:
    """A cnn-mode classifier whose _predict_cnn plays back `outputs` in order."""
    clf = EmotionClassifier(model_path="__no_existe__.pt")
    clf.mode = "cnn"
    queue = list(outputs)
    clf._predict_cnn = lambda face_image: _cnn_output(queue.pop(0))
    return clf


def _calibrate(clf: EmotionClassifier, n: int, blendshapes=None) -> bool:
    for _ in range(n):
        clf.record_calibration_frame(blendshapes, FACE)
    return clf.finalize_baseline()


# A resting face the model reads as mostly neutral with a sad undertone.
RESTING = _probs(neutral=0.70, sad=0.20)


def test_neutral_profile_recovers_sad_masked_by_resting_face():
    clf = _classifier([RESTING] * 12)
    _calibrate(clf, 12, blendshapes={"jawOpen": 0.1})

    raw = _probs(neutral=0.50, sad=0.35)
    assert max(raw, key=raw.get) == "neutral"  # what an uncorrected reading says

    corrected = clf._correct_cnn_probs(raw)
    assert max(corrected, key=corrected.get) == "sad"


def test_face_at_rest_falls_back_to_raw_probs():
    clf = _classifier([RESTING] * 12)
    _calibrate(clf, 12, blendshapes={"jawOpen": 0.1})

    assert clf._correct_cnn_probs(dict(RESTING)) == RESTING


def test_too_few_calibration_frames_leaves_cnn_uncorrected():
    clf = _classifier([RESTING] * 3)
    _calibrate(clf, 3, blendshapes={"jawOpen": 0.1})

    raw = _probs(neutral=0.50, sad=0.35)
    assert clf._cnn_baseline is None
    assert clf._correct_cnn_probs(raw) == raw


def test_cnn_profile_is_set_even_without_blendshapes():
    clf = _classifier([RESTING] * 12)

    has_blendshape_baseline = _calibrate(clf, 12, blendshapes=None)

    assert has_blendshape_baseline is False
    assert clf._cnn_baseline is not None
    assert clf._cnn_baseline.shape == (len(FER7_LABELS),)
    assert clf._cnn_baseline[FER7_LABELS.index("neutral")] == pytest.approx(0.70)
    assert clf._cnn_baseline[FER7_LABELS.index("sad")] == pytest.approx(0.20)
    assert clf._cnn_calibration == []  # emptied once the profile is built


def test_reset_baseline_clears_cnn_profile():
    clf = _classifier([RESTING] * 14)
    _calibrate(clf, 12, blendshapes={"jawOpen": 0.1})
    assert clf._cnn_baseline is not None
    clf.record_calibration_frame(None, FACE)  # a frame of the next, unfinished calibration
    assert len(clf._cnn_calibration) == 1

    clf.reset_baseline()

    assert clf._cnn_baseline is None
    assert clf._cnn_calibration == []
    raw = _probs(neutral=0.50, sad=0.35)
    assert clf._correct_cnn_probs(raw) == raw


def test_corrected_probs_sum_to_one_over_fer7():
    clf = _classifier([RESTING] * 12)
    _calibrate(clf, 12, blendshapes={"jawOpen": 0.1})

    corrected = clf._correct_cnn_probs(_probs(neutral=0.50, sad=0.35))

    assert set(corrected) == set(FER7_LABELS)
    assert sum(corrected.values()) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in corrected.values())


def test_calibration_ignores_face_when_not_cnn_mode():
    def must_not_run(face_image):
        raise AssertionError("_predict_cnn must not run outside cnn mode")

    clf = EmotionClassifier(model_path="__no_existe__.pt")
    clf.mode = "blendshape"
    clf._predict_cnn = must_not_run

    for _ in range(12):
        clf.record_calibration_frame({"jawOpen": 0.1}, FACE)

    assert clf._cnn_calibration == []
    assert clf.finalize_baseline() is True  # blendshape baseline is unaffected
    assert clf._cnn_baseline is None


def test_calibration_without_a_face_crop_records_no_cnn_frame():
    def must_not_run(face_image):
        raise AssertionError("no face crop, nothing to classify")

    clf = EmotionClassifier(model_path="__no_existe__.pt")
    clf.mode = "cnn"
    clf._predict_cnn = must_not_run

    clf.record_calibration_frame({"jawOpen": 0.1})
    clf.record_calibration_frame({"jawOpen": 0.1}, None)

    assert clf._cnn_calibration == []


def test_predict_reports_the_corrected_emotion():
    raw = _probs(neutral=0.50, sad=0.35)
    clf = _classifier([RESTING] * 12 + [raw])
    _calibrate(clf, 12, blendshapes={"jawOpen": 0.1})

    result = clf.predict(face_image=FACE)

    assert result["probabilities"]["sad"] > result["probabilities"]["neutral"]
    assert result["emotion"] == "sad"


def test_predict_without_a_profile_is_uncorrected():
    raw = _probs(neutral=0.50, sad=0.35)
    clf = _classifier([raw])

    result = clf.predict(face_image=FACE)

    assert result["probabilities"]["neutral"] == pytest.approx(raw["neutral"])
    assert result["probabilities"]["sad"] == pytest.approx(raw["sad"])


def test_peak_detection_still_sees_the_uncorrected_probabilities():
    # This person's resting face reads 60% happy. The correction turns the
    # 62% happy reading into a sad frame, but the 0.60 peak threshold was tuned
    # on raw model output, so the peak must still be reported from the raw one.
    resting_smile = _probs(happy=0.60, neutral=0.30)
    raw = _probs(happy=0.62, sad=0.28)
    clf = _classifier([resting_smile] * 12 + [raw])
    _calibrate(clf, 12, blendshapes={"jawOpen": 0.1})

    result = clf.predict(face_image=FACE)

    assert result["emotion"] == "sad"
    assert result["peak"] is not None
    assert result["peak"]["emotion"] == "happy"
    assert result["peak"]["confidence"] == pytest.approx(0.62, abs=1e-3)
