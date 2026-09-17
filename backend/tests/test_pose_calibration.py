"""Tests para la calibración guiada por pose de cabeza."""

import numpy as np
import pytest

from backend.app.services.action_units import ActionUnitAnalyzer
from backend.app.services.pose_calibration import (
    POSE_ANCHORS,
    POSE_SEQUENCE,
    PoseBaselines,
    PoseCalibrationSession,
    pose_matches,
)


def _aus(brow=0.1, mouth=0.2, eye=0.3):
    return {"AU04_brow_lowerer": brow, "AU12_lip_corner": mouth, "AU06_cheek_raiser": eye}


def _run_sequence(session, per_pose_values=None, timestamp_step=0.05):
    """Completa la secuencia entregando frames alineados con cada pose."""
    t = 0.0
    while not session.complete:
        pose = session.current_pose
        assert pose is not None
        yaw, pitch = POSE_ANCHORS[pose]
        values = (per_pose_values or {}).get(pose, _aus())
        session.record(values, {"browDownLeft": values["AU04_brow_lowerer"]}, yaw, pitch, t)
        t += timestamp_step
    return session


# ── Secuencia guiada ──────────────────────────────────────────────────


def test_sequence_starts_on_the_frontal_pose():
    # El centro alimenta la línea base tradicional, así que debe capturarse
    # antes de pedir ningún giro.
    assert POSE_SEQUENCE[0] == "center"
    assert PoseCalibrationSession().current_pose == "center"


def test_samples_are_rejected_while_the_head_is_not_in_the_requested_pose():
    session = PoseCalibrationSession(samples_per_pose=5)

    for i in range(20):
        progress = session.record(_aus(), None, yaw=40.0, pitch=0.0, timestamp=i * 0.05)

    assert progress.aligned is False
    assert progress.samples == 0
    assert session.current_pose == "center"


def test_pose_advances_once_enough_aligned_samples_arrive():
    session = PoseCalibrationSession(samples_per_pose=3)

    for i in range(3):
        session.record(_aus(), None, yaw=0.0, pitch=0.0, timestamp=i * 0.05)

    assert session.current_pose == POSE_SEQUENCE[1]


def test_full_sequence_completes_and_covers_every_pose():
    session = _run_sequence(PoseCalibrationSession(samples_per_pose=4))

    assert session.complete
    assert session.current_pose is None
    assert set(session.build_baselines().calibrated_poses) == set(POSE_SEQUENCE)


def test_timeout_only_reports_after_the_limit_on_an_unfinished_pose():
    session = PoseCalibrationSession(samples_per_pose=5, seconds_per_pose_limit=10.0)
    session.record(_aus(), None, yaw=40.0, pitch=0.0, timestamp=100.0)

    assert session.timed_out(105.0) is False
    assert session.timed_out(111.0) is True


def test_skipping_a_pose_advances_without_recording_it():
    session = PoseCalibrationSession(samples_per_pose=4)
    session.skip_current_pose()

    assert session.current_pose == POSE_SEQUENCE[1]
    assert "center" not in session.build_baselines().calibrated_poses


def test_center_samples_are_exposed_for_the_frontal_baseline():
    session = PoseCalibrationSession(samples_per_pose=3)
    for i in range(3):
        session.record(_aus(brow=0.4), {"browDownLeft": 0.4}, 0.0, 0.0, i * 0.05)

    assert len(session.center_samples()) == 3
    assert len(session.center_blendshapes()) == 3


# ── Tolerancia de pose ────────────────────────────────────────────────


def test_center_demands_a_genuinely_frontal_head():
    assert pose_matches("center", 0.0, 0.0)
    assert not pose_matches("center", 20.0, 0.0)


def test_turned_poses_accept_a_comfortable_margin():
    yaw, pitch = POSE_ANCHORS["right"]
    assert pose_matches("right", yaw, pitch)
    assert pose_matches("right", yaw - 8.0, pitch + 4.0)
    assert not pose_matches("right", yaw - 30.0, pitch)


# ── Interpolación entre poses ─────────────────────────────────────────


def _baselines():
    # El escorzo sube AU04 al girar a la derecha; el resto no cambia, para que
    # la corrección sea comprobable a mano.
    return PoseBaselines(
        action_units={
            "center": {"AU04_brow_lowerer": 0.10, "AU12_lip_corner": 0.20},
            "right": {"AU04_brow_lowerer": 0.30, "AU12_lip_corner": 0.20},
            "left": {"AU04_brow_lowerer": 0.30, "AU12_lip_corner": 0.20},
        }
    )


def test_weights_sum_to_one_and_peak_on_the_nearest_anchor():
    weights = _baselines().weights(*POSE_ANCHORS["right"])

    assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0
    assert max(weights, key=weights.get) == "right"


def test_correction_removes_the_pose_component_at_a_calibrated_anchor():
    baselines = _baselines()
    yaw, pitch = POSE_ANCHORS["right"]

    # Una cara en reposo girada a la derecha mide AU04 = 0.30 por escorzo.
    corrected = baselines.to_frontal({"AU04_brow_lowerer": 0.30}, yaw, pitch)

    # Debe volver al nivel de reposo frontal, no leerse como ceño fruncido.
    assert corrected["AU04_brow_lowerer"] == pytest.approx(0.10, abs=0.03)


def test_a_real_expression_survives_the_correction():
    baselines = _baselines()
    yaw, pitch = POSE_ANCHORS["right"]

    # Mismo giro, pero con 0.25 de ceño real encima del reposo de esa pose.
    corrected = baselines.to_frontal({"AU04_brow_lowerer": 0.55}, yaw, pitch)

    assert corrected["AU04_brow_lowerer"] - 0.10 == pytest.approx(0.25, abs=0.03)


def test_frontal_frames_are_left_untouched():
    baselines = _baselines()
    aus = {"AU04_brow_lowerer": 0.42, "AU12_lip_corner": 0.13}

    corrected = baselines.to_frontal(aus, 0.0, 0.0)

    assert corrected["AU04_brow_lowerer"] == pytest.approx(0.42, abs=0.02)
    assert corrected["AU12_lip_corner"] == pytest.approx(0.13, abs=0.02)


def test_correction_is_continuous_across_the_angle_range():
    # El motivo de interpolar en vez de elegir la pose más cercana: un salto en
    # la línea base aparece aguas abajo como una microexpresión inventada.
    baselines = _baselines()
    resting = 0.30

    values = [
        baselines.to_frontal({"AU04_brow_lowerer": resting}, yaw, 0.0)["AU04_brow_lowerer"]
        for yaw in np.arange(-25.0, 25.5, 0.5)
    ]

    jumps = np.abs(np.diff(values))
    assert jumps.max() < 0.02


def test_uncalibrated_baselines_pass_frames_through_unchanged():
    aus = _aus()

    assert PoseBaselines().to_frontal(aus, 20.0, 0.0) == aus
    only_center = PoseBaselines(action_units={"center": _aus()})
    assert only_center.usable is False
    assert only_center.to_frontal(aus, 20.0, 0.0) == aus


def test_frontal_baseline_ignores_the_buffer_filled_during_the_turns():
    # compute() buffers every frame while no baseline exists, so by the end of
    # the guided sequence that buffer also holds the turned poses. Averaging
    # those into the resting face is the contamination this feature removes.
    analyzer = ActionUnitAnalyzer()
    analyzer._baseline_buffer = [{"AU04_brow_lowerer": 0.9}] * 10

    analyzer.set_baseline([{"AU04_brow_lowerer": 0.1}, {"AU04_brow_lowerer": 0.2}])

    assert analyzer.baseline_set
    assert analyzer.baseline_aus["AU04_brow_lowerer"] == pytest.approx(0.15)
    assert analyzer._baseline_buffer == []


def test_extreme_angles_fall_back_to_the_frontal_baseline():
    baselines = _baselines()

    # Muy fuera del alcance de toda ancla: no debe extrapolar una corrección
    # enorme sin datos que la respalden.
    corrected = baselines.to_frontal({"AU04_brow_lowerer": 0.30}, 500.0, 500.0)

    assert corrected["AU04_brow_lowerer"] == pytest.approx(0.30, abs=1e-6)
