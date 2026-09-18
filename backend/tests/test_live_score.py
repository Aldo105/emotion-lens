import pytest

from backend.app.services.live_score import (
    LiveScoreTracker,
    emotion_valence,
)


# ── Valencia ──────────────────────────────────────────────────────────

def test_valencia_alegria_es_maxima():
    assert emotion_valence({"happy": 1.0}) == pytest.approx(1.0)


def test_valencia_enojo_es_minima():
    assert emotion_valence({"angry": 1.0}) == pytest.approx(-1.0)


def test_valencia_neutral_es_cero():
    assert emotion_valence({"neutral": 1.0}) == pytest.approx(0.0)


def test_tristeza_y_enojo_bajan_respecto_a_neutral():
    neutral = emotion_valence({"neutral": 1.0})
    assert emotion_valence({"sad": 1.0}) < neutral
    assert emotion_valence({"angry": 1.0}) < neutral


def test_alegria_y_confianza_suben_respecto_a_neutral():
    neutral = emotion_valence({"neutral": 1.0})
    assert emotion_valence({"happy": 1.0}) > neutral
    assert emotion_valence({"confidence": 1.0}) > neutral


def test_usa_la_distribucion_no_solo_el_argmax():
    # Misma emoción dominante, reparto distinto: la valencia debe distinguirlas.
    rotunda = emotion_valence({"happy": 0.9, "neutral": 0.1})
    repartida = emotion_valence({"happy": 0.5, "sad": 0.5})
    assert rotunda > repartida


def test_probabilidades_sin_normalizar_no_desbordan():
    v = emotion_valence({"happy": 5.0, "angry": 5.0})
    assert -1.0 <= v <= 1.0
    assert v == pytest.approx(0.0)


def test_distribucion_vacia_es_neutral():
    assert emotion_valence({}) == 0.0
    assert emotion_valence({"happy": 0.0, "sad": 0.0}) == 0.0


def test_etiqueta_desconocida_no_rompe():
    assert emotion_valence({"algo_nuevo": 1.0}) == pytest.approx(0.0)


# ── Tracker ───────────────────────────────────────────────────────────

def _alimentar(tracker, probs, congruence, desde=0.0, hasta=5.0, paso=0.05):
    resultado = None
    t = desde
    while t <= hasta:
        resultado = tracker.update(probs, congruence, t)
        t += paso
    return resultado


def test_no_reporta_antes_del_minimo():
    t = LiveScoreTracker()
    r = t.update({"happy": 1.0}, 80.0, 0.0)
    assert r.score is None
    assert not r.ready


def test_reporta_pasado_el_minimo():
    t = LiveScoreTracker()
    r = _alimentar(t, {"happy": 1.0}, 80.0)
    assert r.ready
    assert r.score is not None


def test_sesion_alegre_puntua_mas_que_una_triste():
    alegre = _alimentar(LiveScoreTracker(), {"happy": 1.0}, 80.0)
    triste = _alimentar(LiveScoreTracker(), {"sad": 1.0}, 80.0)
    assert alegre.score > triste.score


def test_enojo_puntua_por_debajo_de_tristeza():
    triste = _alimentar(LiveScoreTracker(), {"sad": 1.0}, 80.0)
    enojado = _alimentar(LiveScoreTracker(), {"angry": 1.0}, 80.0)
    assert enojado.score < triste.score


def test_mejor_congruencia_sube_el_indice():
    baja = _alimentar(LiveScoreTracker(), {"neutral": 1.0}, 20.0)
    alta = _alimentar(LiveScoreTracker(), {"neutral": 1.0}, 90.0)
    assert alta.score > baja.score


def test_score_siempre_dentro_de_rango():
    for probs, cong in [
        ({"angry": 1.0}, 0.0),
        ({"happy": 1.0}, 100.0),
        ({"sad": 1.0}, -50.0),
        ({"happy": 1.0}, 500.0),
    ]:
        r = _alimentar(LiveScoreTracker(), probs, cong)
        assert 0.0 <= r.score <= 100.0


def test_sin_congruencia_usa_solo_valencia():
    r = _alimentar(LiveScoreTracker(), {"neutral": 1.0}, None)
    assert r.score == pytest.approx(50.0, abs=0.5)
    assert r.congruence is None


def test_la_ventana_olvida_lo_viejo():
    # Arranca triste y luego se pone alegre por más que la ventana: el índice
    # debe describir el presente, no arrastrar toda la sesión.
    t = LiveScoreTracker(smoothing_seconds=3.0)
    _alimentar(t, {"sad": 1.0}, 50.0, desde=0.0, hasta=10.0)
    r = _alimentar(t, {"happy": 1.0}, 50.0, desde=10.05, hasta=20.0)
    solo_alegre = _alimentar(LiveScoreTracker(), {"happy": 1.0}, 50.0)
    assert r.score == pytest.approx(solo_alegre.score, abs=1.0)


def test_reset_vuelve_a_empezar():
    t = LiveScoreTracker()
    _alimentar(t, {"happy": 1.0}, 80.0)
    t.reset()
    r = t.update({"happy": 1.0}, 80.0, 100.0)
    assert not r.ready
    assert r.score is None


def test_muestras_se_acotan_a_la_ventana():
    t = LiveScoreTracker(smoothing_seconds=1.0)
    r = _alimentar(t, {"neutral": 1.0}, 50.0, desde=0.0, hasta=10.0, paso=0.05)
    # 1 s de ventana a 20 Hz ≈ 21 muestras; nunca las 200 de la sesión entera.
    assert r.samples <= 25


def test_as_dict_redondea_y_conserva_none():
    t = LiveScoreTracker()
    d = t.update({"happy": 1.0}, 80.0, 0.0).as_dict()
    assert d["score"] is None
    assert d["ready"] is False
    d2 = _alimentar(t, {"happy": 1.0}, 80.0, desde=0.05, hasta=5.0).as_dict()
    assert isinstance(d2["score"], float)
