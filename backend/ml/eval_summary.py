# backend/ml/eval_summary.py
"""
Resumen del arnés CREMA-D y criterios de promoción de un candidato.

Los criterios se fijan aquí, antes de ver ningún resultado, para que elegir
el modelo no se convierta en buscar el número que convenga.

Uso:
    python -m backend.ml.eval_summary BASE.json CANDIDATO.json
"""
import json
import sys
from collections import Counter, defaultdict

MIN_CLIPS_POR_ACTOR = 5

# Criterios de promoción (fijados el 2026-09-28, antes de entrenar nada)
MEJORA_GLOBAL_MIN = 0.05      # +5 puntos sobre la línea base
MEJORA_SAD_MIN = 0.15         # el objetivo acordado: que 'sad' se detecte
HAPPY_MIN = 0.85              # no romper la única clase que ya funciona
BRECHA_SEXO_TOLERANCIA = 0.02 # la brecha mujeres/hombres no puede crecer


def _tasas(pares):
    return {k: {"aciertos": ok, "total": n, "tasa": round(ok / n, 4)}
            for k, (ok, n) in pares.items()}


def resumir(registros: list[dict]) -> dict:
    grupos = {c: defaultdict(lambda: [0, 0]) for c in ("real", "sexo", "etnia", "actor")}
    confusion = defaultdict(Counter)
    for r in registros:
        acierto = int(r["pred"] == r["real"])
        confusion[r["real"]][r["pred"]] += 1
        for campo, tabla in grupos.items():
            tabla[r[campo]][0] += acierto
            tabla[r[campo]][1] += 1

    n = len(registros)
    aciertos = sum(v[0] for v in grupos["real"].values())
    por_actor = [ok / t for ok, t in grupos["actor"].values() if t >= MIN_CLIPS_POR_ACTOR]
    por_sexo = _tasas(grupos["sexo"])
    tasas_sexo = [v["tasa"] for v in por_sexo.values()]
    return {
        "n": n,
        "global": round(aciertos / n, 4) if n else 0.0,
        "por_emocion": _tasas(grupos["real"]),
        "por_sexo": por_sexo,
        "por_etnia": _tasas(grupos["etnia"]),
        "actores": {
            "peor": round(min(por_actor), 4) if por_actor else 0.0,
            "mejor": round(max(por_actor), 4) if por_actor else 0.0,
            "media": round(sum(por_actor) / len(por_actor), 4) if por_actor else 0.0,
        },
        "brecha_sexo": round(max(tasas_sexo) - min(tasas_sexo), 4) if tasas_sexo else 0.0,
        "confusion": {real: dict(c) for real, c in confusion.items()},
    }


def _tasa(res, emocion):
    return res["por_emocion"].get(emocion, {}).get("tasa", 0.0)


def cumple_criterios(base: dict, cand: dict) -> tuple[bool, list[str]]:
    fallos = []
    if cand["global"] < base["global"] + MEJORA_GLOBAL_MIN:
        fallos.append(f"global {cand['global']:.2f} < {base['global']:.2f} + {MEJORA_GLOBAL_MIN}")
    if _tasa(cand, "sad") < _tasa(base, "sad") + MEJORA_SAD_MIN:
        fallos.append(f"sad {_tasa(cand, 'sad'):.2f} < {_tasa(base, 'sad'):.2f} + {MEJORA_SAD_MIN}")
    if _tasa(cand, "happy") < HAPPY_MIN:
        fallos.append(f"happy {_tasa(cand, 'happy'):.2f} < {HAPPY_MIN}")
    if cand["actores"]["peor"] < base["actores"]["peor"]:
        fallos.append(f"peor actor {cand['actores']['peor']:.2f} < {base['actores']['peor']:.2f}")
    if cand["brecha_sexo"] > base["brecha_sexo"] + BRECHA_SEXO_TOLERANCIA:
        fallos.append(f"brecha sexo {cand['brecha_sexo']:.2f} > {base['brecha_sexo']:.2f} + {BRECHA_SEXO_TOLERANCIA}")
    return (not fallos, fallos)


def main(argv):
    base, cand = (json.loads(open(p, encoding="utf-8").read()) for p in argv[:2])
    for clave in ("global", "brecha_sexo"):
        print(f"{clave:<12} {base[clave]:.2f} -> {cand[clave]:.2f}")
    print(f"{'peor actor':<12} {base['actores']['peor']:.2f} -> {cand['actores']['peor']:.2f}")
    for emo in sorted(base["por_emocion"]):
        print(f"{emo:<12} {_tasa(base, emo):.2f} -> {_tasa(cand, emo):.2f}")
    ok, fallos = cumple_criterios(base, cand)
    print("\nPROMOCIONABLE" if ok else "\nNO PROMOCIONABLE:\n  " + "\n  ".join(fallos))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
