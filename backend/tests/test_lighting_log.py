from backend.app.routers.websocket import LightingLog


def _run(capsys, frames):
    log = LightingLog(session_id=7)
    for t, state, brightness in frames:
        log.update(state, t, brightness)
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("[LUZ]")]


def test_reports_low_light_once_it_holds(capsys):
    frames = [(t / 10, "poca_luz", 0.33) for t in range(0, 40)]
    lines = _run(capsys, frames)
    assert lines == ["[LUZ] Sesión 7 (2s): poca luz, no se muestran emociones (brillo 0.33)"]


def test_flicker_around_the_cut_is_not_reported(capsys):
    # Brightness bouncing across 0.40 every half second never holds 2 s.
    frames = [(t / 10, "poca_luz" if (t // 5) % 2 else "ok", 0.4) for t in range(0, 60)]
    assert _run(capsys, frames) == []


def test_recovery_is_reported_but_a_good_start_is_not(capsys):
    frames = ([(t / 10, "ok", 0.5) for t in range(0, 30)]
              + [(3 + t / 10, "poca_luz", 0.3) for t in range(0, 30)]
              + [(6 + t / 10, "ok", 0.5) for t in range(0, 30)])
    lines = _run(capsys, frames)
    assert len(lines) == 2
    assert "poca luz" in lines[0]
    assert "luz suficiente" in lines[1]
