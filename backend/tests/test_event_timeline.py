"""Tests for the moderator event timeline and per-task friction analysis."""

import numpy as np

from backend.app.services.interview_analyzer import InterviewBehaviorAnalyzer


def _series(duration_s=60.0, step=0.5, nervousness_fn=None, congruence_fn=None,
            emotion_fn=None):
    """Build synthetic per-frame arrays for a session."""
    timestamps = np.arange(0.0, duration_s, step)
    nervousness = np.array([
        nervousness_fn(t) if nervousness_fn else 0.2 for t in timestamps
    ])
    congruence = np.array([
        congruence_fn(t) if congruence_fn else 80.0 for t in timestamps
    ])
    emotions = [emotion_fn(t) if emotion_fn else "neutral" for t in timestamps]
    return timestamps, emotions, nervousness, congruence


def _analyzer():
    return InterviewBehaviorAnalyzer()


def test_event_timeline_ignores_untagged_notes():
    ts, emo, nerv, cong = _series()
    notes = [
        {"timestamp": 20.0, "content": "just a thought", "tag": None},
        {"timestamp": 25.0, "content": "unknown tag", "tag": "something_else"},
    ]

    events = _analyzer()._build_event_timeline(ts, emo, nerv, cong, notes)

    assert events == []


def test_event_timeline_flags_nervousness_spike_as_friction():
    # Calm before t=30, nervous after
    ts, emo, nerv, cong = _series(
        nervousness_fn=lambda t: 0.1 if t < 30 else 0.7
    )
    notes = [{"timestamp": 30.0, "content": "clicked wrong button", "tag": "error"}]

    events = _analyzer()._build_event_timeline(ts, emo, nerv, cong, notes)

    assert len(events) == 1
    event = events[0]
    assert event["is_friction_point"] is True
    assert event["nervousness_change"] > 0.5
    assert event["video_time"] == "00:30"
    assert "friccion" in event["interpretation"]


def test_event_timeline_reports_no_change_when_state_is_stable():
    ts, emo, nerv, cong = _series()
    notes = [{"timestamp": 30.0, "content": "asked about salary", "tag": "key_question"}]

    events = _analyzer()._build_event_timeline(ts, emo, nerv, cong, notes)

    assert events[0]["is_friction_point"] is False
    assert events[0]["nervousness_change"] == 0.0


def test_manual_error_marker_is_friction_even_without_emotional_reaction():
    """The moderator saw a failure; a flat face doesn't make it not a failure."""
    ts, emo, nerv, cong = _series()
    notes = [{"timestamp": 30.0, "content": "could not find the menu", "tag": "error"}]

    events = _analyzer()._build_event_timeline(ts, emo, nerv, cong, notes)

    assert events[0]["is_friction_point"] is True


def test_event_outside_the_recorded_window_is_skipped():
    """A marker dropped during calibration has no frames around it."""
    ts, emo, nerv, cong = _series(duration_s=60.0)
    notes = [{"timestamp": 500.0, "content": "late marker", "tag": "error"}]

    events = _analyzer()._build_event_timeline(ts, emo, nerv, cong, notes)

    assert events == []


def test_events_are_returned_in_chronological_order():
    ts, emo, nerv, cong = _series()
    notes = [
        {"timestamp": 40.0, "content": "third", "tag": "error"},
        {"timestamp": 10.0, "content": "first", "tag": "confusion"},
        {"timestamp": 25.0, "content": "second", "tag": "key_question"},
    ]

    events = _analyzer()._build_event_timeline(ts, emo, nerv, cong, notes)

    assert [e["content"] for e in events] == ["first", "second", "third"]


def test_task_analysis_pairs_start_with_next_end():
    ts, emo, nerv, cong = _series(duration_s=120.0)
    notes = [
        {"timestamp": 10.0, "content": "Completar registro", "tag": "task_start"},
        {"timestamp": 40.0, "content": "listo", "tag": "task_end"},
        {"timestamp": 50.0, "content": "Buscar producto", "tag": "task_start"},
        {"timestamp": 80.0, "content": "listo", "tag": "task_end"},
    ]

    tasks = _analyzer()._build_task_analysis(ts, emo, nerv, cong, notes)

    assert len(tasks) == 2
    assert tasks[0]["task_name"] == "Completar registro"
    assert tasks[0]["completed"] is True
    assert tasks[0]["duration_seconds"] == 30.0
    assert tasks[0]["start_video_time"] == "00:10"
    assert tasks[1]["task_name"] == "Buscar producto"


def test_unfinished_task_runs_to_session_end_and_is_marked_incomplete():
    ts, emo, nerv, cong = _series(duration_s=100.0)
    notes = [{"timestamp": 20.0, "content": "Tarea abandonada", "tag": "task_start"}]

    tasks = _analyzer()._build_task_analysis(ts, emo, nerv, cong, notes)

    assert len(tasks) == 1
    assert tasks[0]["completed"] is False
    assert tasks[0]["end"] > 90.0
    assert "sin marcar como completada" in tasks[0]["interpretation"]


def test_friction_score_rises_with_nervousness_and_marked_errors():
    ts, emo, nerv, cong = _series(
        duration_s=120.0,
        nervousness_fn=lambda t: 0.8 if 10 <= t <= 40 else 0.1,
    )
    calm_notes = [
        {"timestamp": 50.0, "content": "Tarea facil", "tag": "task_start"},
        {"timestamp": 80.0, "content": "listo", "tag": "task_end"},
    ]
    hard_notes = [
        {"timestamp": 10.0, "content": "Tarea dificil", "tag": "task_start"},
        {"timestamp": 20.0, "content": "no encontro el boton", "tag": "error"},
        {"timestamp": 25.0, "content": "dudo mucho", "tag": "confusion"},
        {"timestamp": 40.0, "content": "listo", "tag": "task_end"},
    ]

    calm = _analyzer()._build_task_analysis(ts, emo, nerv, cong, calm_notes)[0]
    hard = _analyzer()._build_task_analysis(ts, emo, nerv, cong, hard_notes)[0]

    assert hard["friction_score"] > calm["friction_score"]
    assert hard["error_count"] == 1
    assert hard["confusion_count"] == 1
    assert hard["friction_score"] <= 100


def test_no_task_markers_yields_no_task_analysis():
    ts, emo, nerv, cong = _series()
    notes = [{"timestamp": 30.0, "content": "just a question", "tag": "key_question"}]

    assert _analyzer()._build_task_analysis(ts, emo, nerv, cong, notes) == []


def test_peak_nervousness_time_is_reported_within_the_task():
    ts, emo, nerv, cong = _series(
        duration_s=120.0,
        nervousness_fn=lambda t: 0.9 if 34.0 <= t <= 36.0 else 0.1,
    )
    notes = [
        {"timestamp": 10.0, "content": "Tarea", "tag": "task_start"},
        {"timestamp": 60.0, "content": "listo", "tag": "task_end"},
    ]

    task = _analyzer()._build_task_analysis(ts, emo, nerv, cong, notes)[0]

    assert task["peak_nervousness"] == 0.9
    assert task["peak_nervousness_time"] == "00:34"
