"""
Tests para la magnificación EVM en vivo (el filtro del toggle).

Se construyen cuadros sintéticos con una oscilación de color conocida en la
banda del pulso y se comprueba que el filtro la amplifica, que ignora lo que
cae fuera de la banda, y que no toca el cuadro cuando no puede trabajar.
"""
import numpy as np
import pytest

from backend.app.services.live_evm import (
    MIN_ROI_PX,
    LiveEVMMagnifier,
    _ema_alpha,
)

FPS = 20.0
FRAME_H, FRAME_W = 240, 320
BBOX = {"x_min": 80, "y_min": 40, "x_max": 240, "y_max": 200}


def _frame(red_level):
    """Cuadro gris uniforme con el canal rojo desplazado."""
    frame = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)
    frame[:, :, 2] = np.clip(128 + red_level, 0, 255)
    return frame


def _run(magnifier, freq_hz, amplitude=4.0, seconds=6.0):
    """Alimenta una oscilación y devuelve la amplitud de salida en el ROI."""
    outputs = []
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        level = amplitude * np.sin(2 * np.pi * freq_hz * t)
        out = magnifier.process(_frame(level), BBOX)
        # Centro del ROI, lejos del borde difuminado y del distintivo.
        outputs.append(float(out[120, 160, 2]))
    # Se descarta el primer tercio: el filtro arranca en cero y necesita
    # asentarse antes de que su salida signifique algo.
    settled = outputs[len(outputs) // 3:]
    return max(settled) - min(settled)


def _magnifier(**kwargs):
    params = dict(
        amplification=40.0, freq_low=0.7, freq_high=3.0,
        pyramid_levels=4, fps=FPS,
    )
    params.update(kwargs)
    return LiveEVMMagnifier(**params)


# ── Coeficiente del filtro ────────────────────────────────────────────


def test_ema_alpha_grows_with_the_cutoff():
    slow = _ema_alpha(0.7, FPS)
    fast = _ema_alpha(3.0, FPS)

    assert 0.0 < slow < fast < 1.0


def test_ema_alpha_is_safe_at_degenerate_inputs():
    assert _ema_alpha(0.0, FPS) == 1.0
    assert _ema_alpha(1.0, 0.0) == 1.0


# ── Comportamiento del filtro ─────────────────────────────────────────


def test_a_pulse_band_oscillation_comes_out_amplified():
    # 1.2 Hz = 72 BPM, en plena banda.
    amplified = _run(_magnifier(), freq_hz=1.2)
    passthrough = _run(_magnifier(amplification=0.0), freq_hz=1.2)

    assert amplified > passthrough * 2


def test_amplification_scales_the_output():
    weak = _run(_magnifier(amplification=10.0), freq_hz=1.2)
    strong = _run(_magnifier(amplification=40.0), freq_hz=1.2)

    assert strong > weak


def test_fast_flicker_above_the_band_is_rejected_more_than_the_pulse():
    # Lo que se mueve rápido en un video de una cara es ruido de sensor y
    # movimiento, no flujo sanguíneo. Con una sola sección IIR (6 dB/octava)
    # salía casi tan amplificado como el pulso y ×40 se veía como grano; la
    # cascada es lo que abre esta diferencia.
    pulse = _run(_magnifier(), freq_hz=1.2, seconds=20.0)
    flicker = _run(_magnifier(), freq_hz=6.0, seconds=20.0)

    assert pulse > flicker * 2


def test_slow_drift_below_the_band_is_not_amplified():
    # Un cambio de iluminación lento no es pulso; amplificarlo llenaría la cara
    # de color al mover una lámpara.
    in_band = _run(_magnifier(), freq_hz=1.2)
    drift = _run(_magnifier(), freq_hz=0.08, seconds=20.0)

    assert drift < in_band


# ── Casos en los que no debe tocar el cuadro ──────────────────────────


def test_frames_pass_through_untouched_while_the_filter_warms_up():
    magnifier = _magnifier()
    frame = _frame(0)

    # Las dos medias móviles nacen iguales, así que su diferencia empieza en
    # cero: amplificar ese transitorio pintaría un destello que no es pulso.
    first = magnifier.process(frame, BBOX)

    assert np.array_equal(first, frame)
    assert magnifier.warmed_up is False


def test_no_face_leaves_the_frame_alone():
    frame = _frame(0)

    assert np.array_equal(_magnifier().process(frame, None), frame)


def test_a_roi_too_small_to_work_with_is_left_alone():
    frame = _frame(0)
    tiny = {"x_min": 10, "y_min": 10,
            "x_max": 10 + MIN_ROI_PX - 1, "y_max": 10 + MIN_ROI_PX - 1}

    assert np.array_equal(_magnifier().process(frame, tiny), frame)


def test_output_keeps_the_frame_shape_and_type():
    magnifier = _magnifier()
    for _ in range(int(FPS * 2)):
        out = magnifier.process(_frame(3), BBOX)

    assert out.shape == (FRAME_H, FRAME_W, 3)
    assert out.dtype == np.uint8


def test_pixels_outside_the_face_are_never_modified():
    magnifier = _magnifier()
    out = None
    for _ in range(int(FPS * 3)):
        frame = _frame(4)
        out = magnifier.process(frame, BBOX)

    # Esquina superior izquierda, fuera del recuadro del rostro.
    assert out[5, 5, 2] == frame[5, 5, 2]


# ── Estado ────────────────────────────────────────────────────────────


def test_reset_puts_the_filter_back_in_its_warm_up_state():
    magnifier = _magnifier()
    for _ in range(int(FPS * 3)):
        magnifier.process(_frame(2), BBOX)
    assert magnifier.warmed_up

    magnifier.reset()

    assert magnifier.warmed_up is False
    frame = _frame(0)
    assert np.array_equal(magnifier.process(frame, BBOX), frame)


def test_a_resized_face_box_does_not_break_the_filter():
    # El recuadro cambia de tamaño con cada cuadro conforme la persona se
    # mueve; el ROI se normaliza antes del filtro, así que el estado sobrevive.
    magnifier = _magnifier()
    for i in range(int(FPS * 3)):
        grow = i % 7
        bbox = {
            "x_min": BBOX["x_min"], "y_min": BBOX["y_min"],
            "x_max": BBOX["x_max"] + grow, "y_max": BBOX["y_max"] + grow,
        }
        out = magnifier.process(_frame(3), bbox)

    assert magnifier.warmed_up
    assert out.shape == (FRAME_H, FRAME_W, 3)
