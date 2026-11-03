import json

import cv2
import numpy as np

from backend.app.services import prediction_video_renderer as pvr


def test_record_at_uses_the_last_record_and_goes_blank_when_stale():
    stamps = [1.0, 1.05, 1.10, 5.0]
    assert pvr.record_at(stamps, 0.5) is None          # before the first record
    assert pvr.record_at(stamps, 1.07) == 1
    assert pvr.record_at(stamps, 1.9) == 2             # 0.8 s old: still current
    assert pvr.record_at(stamps, 3.0) is None          # frames dropped (low light): no reading
    assert pvr.record_at(stamps, 5.2) == 3


def _video(path, frames=40, fps=20.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for i in range(frames):
        w.write(np.full((480, 640, 3), i * 5 % 255, np.uint8))
    w.release()


def test_renders_side_by_side_video_and_reports_ready(tmp_path):
    src = tmp_path / "src.mp4"
    _video(src)
    out, prog = pvr.output_paths(str(tmp_path), 7)
    records = [{"timestamp": t / 20, "emotion": "happy" if t < 20 else "sad",
                "emotion_probabilities": {"happy": 0.7, "sad": 0.2}} for t in range(40)]
    result = pvr.render_prediction_video(7, str(src), records, [{"timestamp": 1.0, "detected_emotion": "stress"}],
                                         out, prog, session_duration=None)
    assert result == out
    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == pvr.WIDTH
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == pvr.HEIGHT
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    assert json.load(open(prog, encoding="utf-8"))["status"] == "ready"
    assert not (tmp_path / "7_prediction.part.mp4").exists()


def test_live_session_time_is_mapped_onto_a_shorter_video(tmp_path):
    # 40 frames at 20 fps = 2 s of video for a 4 s session: the record at
    # session second 3.9 must still be reached near the end of the video.
    src = tmp_path / "src.mp4"
    _video(src)
    out, prog = pvr.output_paths(str(tmp_path), 8)
    records = [{"timestamp": 3.9, "emotion": "angry", "emotion_probabilities": {"angry": 0.9}}]
    assert pvr.render_prediction_video(8, str(src), records, [], out, prog, session_duration=4.0) == out


def test_unreadable_source_marks_the_render_failed(tmp_path):
    out, prog = pvr.output_paths(str(tmp_path), 9)
    assert pvr.render_prediction_video(9, str(tmp_path / "missing.mp4"), [], [], out, prog) is None
    assert json.load(open(prog, encoding="utf-8"))["status"] == "failed"
    assert not (tmp_path / "9_prediction.mp4").exists()
