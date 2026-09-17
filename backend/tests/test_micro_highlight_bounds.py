"""
Tests para la seleccion de frames del video de validacion de microexpresiones.

El grabador declara el MP4 a 20 FPS, pero el navegador entrega frames en un
intervalo best-effort, asi que la tasa real es menor. Estas pruebas fijan que
el recorte se ubique por los timestamps reales de cada frame y no por la tasa
nominal, que era lo que descartaba los clips tardios.
"""
import numpy as np

from backend.app.services.micro_expression_highlight_renderer import (
    MIN_CLIP_FRAMES,
    _clip_bounds,
)

NOMINAL_FPS = 20.0
PADDING = 0.75


def _recording(duration_s, real_fps):
    """Timestamps de una grabacion entregada a `real_fps` durante `duration_s`."""
    return np.arange(0.0, duration_s, 1.0 / real_fps, dtype=np.float64)


def test_clip_is_centred_on_the_event_at_the_real_frame_rate():
    ts = _recording(180.0, real_fps=14.0)
    event_t = 120.0

    start, end = _clip_bounds(ts, event_t, PADDING, len(ts))

    # El clip debe cubrir el evento, no una ventana desplazada.
    assert ts[start] <= event_t <= ts[end - 1]
    assert abs(ts[start] - (event_t - PADDING)) < 0.1
    assert abs(ts[end - 1] - (event_t + PADDING)) < 0.1


def test_late_event_is_not_dropped_when_capture_runs_below_nominal_fps():
    # 3 minutos entregados a 14 FPS: el video tiene ~2520 frames, pero la
    # aritmetica nominal ubicaba un evento de t=150s en el frame 3000, fuera
    # del archivo, y el clip se descartaba por "demasiado corto".
    ts = _recording(180.0, real_fps=14.0)
    event_t = 150.0

    nominal_start = int(round(event_t * NOMINAL_FPS)) - int(round(PADDING * NOMINAL_FPS))
    assert nominal_start > len(ts), "el caso de regresion debe quedar fuera del video"

    start, end = _clip_bounds(ts, event_t, PADDING, len(ts))

    assert end - start >= MIN_CLIP_FRAMES


def test_event_past_the_end_of_the_recording_still_yields_an_empty_range():
    ts = _recording(60.0, real_fps=14.0)

    start, end = _clip_bounds(ts, 120.0, PADDING, len(ts))

    assert end - start < MIN_CLIP_FRAMES


def test_bounds_stay_inside_the_video_when_the_event_sits_at_the_edges():
    ts = _recording(60.0, real_fps=14.0)

    start, end = _clip_bounds(ts, 0.0, PADDING, len(ts))
    assert start == 0
    assert end <= len(ts)

    start, end = _clip_bounds(ts, float(ts[-1]), PADDING, len(ts))
    assert end <= len(ts)


def test_total_frames_caps_the_range_when_it_is_shorter_than_the_metadata():
    # CAP_PROP_FRAME_COUNT puede quedar por debajo de los frames escritos si el
    # contenedor se cerro a medias; el rango no debe salirse del archivo.
    ts = _recording(60.0, real_fps=14.0)
    truncated = 100

    start, end = _clip_bounds(ts, 30.0, PADDING, truncated)

    assert end <= truncated
