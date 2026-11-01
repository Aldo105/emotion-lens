import time

from backend.app.services.emotion_classifier import EmotionClassifier


def _probs(**kw):
    base = {"neutral": 0.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, "surprise": 0.0,
            "disgust": 0.0, "fear": 0.0, "nervousness": 0.0, "confidence": 0.0}
    base.update(kw)
    return base


def _hold(clf, probs, seconds):
    end = time.time() + seconds
    while time.time() < end:
        clf._prob_history.append(probs)
        emotion, conf = clf._apply_hysteresis(probs)
        time.sleep(0.02)
    return emotion, conf


def test_a_label_with_no_support_left_is_not_kept_by_the_anti_bounce_check():
    # neutral -> happy, then straight back to neutral within the 3 s bounce
    # window. The live view showed "Feliz 0%" here: the label stuck on an
    # emotion whose probability had dropped to zero.
    clf = EmotionClassifier(device="cpu")
    clf._current_emotion = "neutral"
    assert _hold(clf, _probs(happy=0.8, neutral=0.2), 0.5)[0] == "happy"

    emotion, conf = _hold(clf, _probs(neutral=0.7, angry=0.2, sad=0.1), 0.5)
    assert emotion == "neutral"
    assert conf > 0.5


def test_a_short_flicker_back_is_still_held_while_the_label_keeps_support():
    # The bounce guard still does its job when the shown emotion is a close
    # runner-up rather than gone.
    clf = EmotionClassifier(device="cpu")
    clf._current_emotion = "neutral"
    _hold(clf, _probs(happy=0.8, neutral=0.2), 0.5)
    emotion, _ = _hold(clf, _probs(neutral=0.5, happy=0.45, sad=0.05), 0.5)
    assert emotion == "happy"
