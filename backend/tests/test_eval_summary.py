# backend/tests/test_eval_summary.py
from backend.ml.eval_summary import comparables, cumple_criterios, resumir


def _reg(actor, real, pred, sexo="Female", etnia="Caucasian"):
    return {"actor": actor, "real": real, "pred": pred, "sexo": sexo, "etnia": etnia}


def _registros():
    regs = []
    for i in range(5):  # actor A: 5/5, actor B: 1/5
        regs.append(_reg("A", "happy", "happy"))
        regs.append(_reg("B", "sad", "sad" if i == 0 else "neutral", sexo="Male"))
    return regs


def test_resumir_global_and_per_actor():
    r = resumir(_registros())
    assert r["n"] == 10
    assert r["global"] == 0.6
    assert r["actores"] == {"peor": 0.2, "mejor": 1.0, "media": 0.6}
    assert r["por_emocion"]["sad"] == {"aciertos": 1, "total": 5, "tasa": 0.2}
    assert r["brecha_sexo"] == 0.8
    assert r["confusion"]["sad"] == {"sad": 1, "neutral": 4}


def test_resumir_ignores_actors_with_fewer_than_5_clips_for_spread():
    regs = _registros() + [_reg("C", "angry", "happy")]
    assert resumir(regs)["actores"]["peor"] == 0.2


def _summary(glob, sad, happy, peor, brecha):
    return {"global": glob, "actores": {"peor": peor}, "brecha_sexo": brecha,
            "por_emocion": {"sad": {"tasa": sad}, "happy": {"tasa": happy}}}


def test_cumple_criterios_accepts_clear_improvement():
    base = _summary(0.36, 0.10, 0.95, 0.09, 0.10)
    cand = _summary(0.45, 0.30, 0.90, 0.15, 0.08)
    assert cumple_criterios(base, cand) == (True, [])


def test_cumple_criterios_lists_every_failure():
    base = _summary(0.36, 0.10, 0.95, 0.09, 0.10)
    cand = _summary(0.38, 0.20, 0.80, 0.05, 0.20)
    ok, fallos = cumple_criterios(base, cand)
    assert not ok
    assert len(fallos) == 5


def test_comparables_rejects_empty_candidate():
    assert comparables({"n": 220}, {"n": 0})


def test_comparables_rejects_different_clip_count():
    assert comparables({"n": 220}, {"n": 200})


def test_comparables_accepts_same_n():
    assert comparables({"n": 220}, {"n": 220}) == []
