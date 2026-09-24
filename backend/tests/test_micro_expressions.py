"""Tests del detector de microexpresiones (onset→apex→offset y etiquetas)."""

import numpy as np
import pytest

from backend.app.services.micro_expressions import (
    MicroExpressionEngine,
    micro_contradicts_displayed,
)

FPS = 20
TRACKED = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU12", "AU14",
           "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26"]
REST = 0.06


class _Ema:
    """El suavizado de ActionUnitAnalyzer (alpha 0.5), que el motor recibe ya aplicado."""

    def __init__(self):
        self.prev = None

    def __call__(self, frame):
        if self.prev is None:
            self.prev = dict(frame)
        else:
            self.prev = {k: 0.5 * v + 0.5 * self.prev[k] for k, v in frame.items()}
        return dict(self.prev)


def _engine():
    engine = MicroExpressionEngine()
    engine.set_baseline(
        {au: REST for au in TRACKED} | {"face_symmetry": 0.95},
        {au: 0.01 for au in TRACKED} | {"face_symmetry": 0.0},
    )
    return engine


class _State:
    def __init__(self, upper_face_only=False, allowed=()):
        self.is_yawning = self.is_scratching = self.is_forced_blink = False
        self.upper_face_only = upper_face_only
        self.filtered_aus_for_micro = set(allowed)


def _feed(engine, seconds, bumps=(), dominant="happy", state_at=None, t0=100.0):
    """Alimenta el motor; `bumps` = [(inicio, duración, {AU: incremento})]."""
    ema, events = _Ema(), []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        frame = {au: REST for au in TRACKED}
        for start, duration, extra in bumps:
            if start <= t < start + duration:
                for au, value in extra.items():
                    frame[au] += value
        frame["face_symmetry"] = 0.95
        state = state_at(t) if state_at else None
        event = engine.analyze(ema(frame), t, dominant_emotion=dominant, noise_state=state)
        if event:
            events.append(event)
    return events


# ── Etiquetas: micro vs. emoción mostrada ────────────────────────────

@pytest.mark.parametrize("micro, shown", [("anger", "angry"), ("sadness", "sad"),
                                          ("disgust", "disgust"), ("surprise", "surprise")])
def test_misma_emocion_con_otro_nombre_no_es_contradiccion(micro, shown):
    assert not micro_contradicts_displayed(micro, shown)


@pytest.mark.parametrize("micro, shown", [("anger", "happy"), ("sadness", "happy"),
                                          ("contempt", "happy"), ("fear", "sad")])
def test_emociones_distintas_si_son_contradiccion(micro, shown):
    assert micro_contradicts_displayed(micro, shown)


@pytest.mark.parametrize("shown", ["neutral", "unmeasurable", None])
def test_sin_emocion_mostrada_no_hay_contradiccion(shown):
    assert not micro_contradicts_displayed("anger", shown)


# ── Detección ────────────────────────────────────────────────────────

@pytest.mark.parametrize("duration", [0.10, 0.15, 0.20, 0.30])
def test_detecta_una_expresion_breve_real(duration):
    """Con el doble EMA anterior ninguna de estas pasaba el filtro temporal."""
    events = _feed(_engine(), 3.0, bumps=[(101.0, duration, {"AU1": 0.35, "AU2": 0.35, "AU4": 0.35})])
    assert len(events) == 1
    assert events[0].detected_emotion == "fear"
    assert 40 <= events[0].duration_ms <= 500


def test_una_expresion_se_reporta_una_sola_vez():
    events = _feed(_engine(), 4.0, bumps=[(101.0, 0.2, {"AU9": 0.35})])
    assert len(events) == 1


def test_expresion_sostenida_no_es_micro():
    events = _feed(_engine(), 4.0, bumps=[(101.0, 1.5, {"AU1": 0.35, "AU4": 0.35, "AU15": 0.35})])
    assert events == []


def test_un_solo_cuadro_no_basta():
    """Un salto de un cuadro no se distingue del ruido de landmarks a 20 fps."""
    events = _feed(_engine(), 3.0, bumps=[(101.0, 0.05, {"AU9": 0.35})])
    assert events == []


def test_aus_que_no_coinciden_en_el_tiempo_no_forman_un_patron():
    # AU1 y AU15 (tristeza) con 1.2 s de separación: son dos movimientos distintos.
    events = _feed(_engine(), 4.0, bumps=[(101.0, 0.2, {"AU1": 0.35}), (102.2, 0.2, {"AU15": 0.35})])
    assert all(e.detected_emotion != "sadness" for e in events)


def test_hablar_no_crea_picos_artificiales_en_la_boca():
    """
    Mientras se detecta habla, las AUs de la boca quedan fuera de la lectura.
    Antes se leían como 0 y el vaivén hablando/no hablando formaba picos falsos.
    """
    upper = {"AU1", "AU2", "AU4", "AU7", "AU9"}

    def flicker(t):  # el estado de habla cambia cada cuadro
        return _State(upper_face_only=int(t * FPS) % 2 == 0, allowed=upper)

    engine = _engine()
    ema, events = _Ema(), []
    for i in range(int(4.0 * FPS)):
        t = 100.0 + i / FPS
        frame = {au: REST for au in TRACKED} | {"AU23": 0.4, "AU24": 0.4, "face_symmetry": 0.95}
        event = engine.analyze(ema(frame), t, dominant_emotion="happy", noise_state=flicker(t))
        if event:
            events.append(event)
    assert events == []


def test_la_emocion_mostrada_equivalente_no_marca_contradiccion():
    events = _feed(_engine(), 3.0, bumps=[(101.0, 0.2, {"AU1": 0.35, "AU4": 0.35, "AU15": 0.35})],
                   dominant="sad")
    assert events and not events[0].is_contradictory


def test_recalibrar_olvida_la_calibracion_anterior():
    engine = MicroExpressionEngine()
    for _ in range(100):
        engine.record_calibration_frame({"AU12": 0.5})
    engine.set_baseline({"AU12": 0.5}, {"AU12": 0.05})
    assert engine._is_habitual("AU12")

    engine.reset_calibration()
    for _ in range(100):
        engine.record_calibration_frame({"AU12": 0.05})
    engine.set_baseline({"AU12": 0.05}, {"AU12": 0.01})
    assert not engine._is_habitual("AU12")
