import numpy as np

from backend.app.config import BASE_DIR
from backend.app.models.schemas import WSFrameResult
from backend.app.services.action_units import ActionUnitAnalyzer

FRONT = BASE_DIR / "frontend"


def _read(rel):
    return (FRONT / rel).read_text(encoding="utf-8")


def _landmarks(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.7, size=(468, 3))
    return [{"x": float(x), "y": float(y), "z": float(z) * 0.1} for x, y, z in pts]


def test_live_analyzer_does_not_buffer_frames_for_a_baseline():
    # Live sessions never set a baseline, so buffering would grow all session.
    au = ActionUnitAnalyzer(collect_baseline=False)
    for i in range(50):
        au.compute(_landmarks(i), (480, 640), timestamp=i / 20)
    assert au._baseline_buffer == []
    assert not au.baseline_set


def test_default_analyzer_still_buffers_for_offline_calibration():
    # video_processor.py still calibrates from the first seconds of a video.
    au = ActionUnitAnalyzer()
    for i in range(5):
        au.compute(_landmarks(i), (480, 640), timestamp=i / 20)
    assert len(au._baseline_buffer) == 5


def test_frame_result_carries_no_calibration_fields():
    fields = WSFrameResult.model_fields
    for name in ("is_calibrating", "calibration_progress", "calibration_pose"):
        assert name not in fields


def test_live_page_has_no_calibration_ui():
    html = _read("index.html")
    assert 'id="calibration-bar"' not in html
    assert 'id="btn-recalibrate"' not in html
    assert "recalibrate" not in _read("js/websocket.js")
    assert "is_calibrating" not in _read("js/dashboard.js")


def test_ending_a_session_shows_no_survey():
    html = _read("index.html")
    assert 'id="survey-modal"' not in html
    assert "post-session-survey.js" not in html
    assert "PostSessionSurvey" not in _read("js/app.js")
    assert not (FRONT / "js" / "post-session-survey.js").exists()
