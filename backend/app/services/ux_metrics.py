"""
EmotionLens — Métricas para pruebas de usabilidad

Deliberadamente distintas de las del análisis de entrevista: aquí no se puntúa
a la persona, se describe la sesión para que un investigador sepa dónde mirar.

Qué se mide y por qué solo esto. Sobre 20 actores distintos el clasificador
acierta un 36% en general, pero un 95% en 'happy' (ver references.py). Así que
las únicas métricas defendibles son las que dependen de la sonrisa, y las que
cuentan *cambios* en la cara sin afirmar qué emoción hubo. Todo lo que exija
nombrar una emoción con precisión queda fuera a propósito.

En una prueba de usabilidad esto basta, porque el sistema no decide nada:
señala momentos de una sesión larga para que alguien los revise, y agrega
entre participantes, donde el ruido individual se promedia.
"""

from collections import Counter

import numpy as np

# Única etiqueta que aguanta un cambio de cara (95% sobre 20 actores)
ETIQUETA_FIABLE = "happy"


def _tramos(registros: list[dict], marcadores: list[dict]) -> list[dict]:
    """Parte la sesión en tramos según las marcas del moderador."""
    if not marcadores:
        if not registros:
            return []
        return [{
            "nombre": "sesión completa",
            "inicio": registros[0]["timestamp"],
            "fin": registros[-1]["timestamp"],
        }]

    ordenados = sorted(marcadores, key=lambda m: m["timestamp"])
    fin_sesion = registros[-1]["timestamp"] if registros else 0.0
    tramos = []
    for i, m in enumerate(ordenados):
        fin = ordenados[i + 1]["timestamp"] if i + 1 < len(ordenados) else fin_sesion
        tramos.append({
            "nombre": (m.get("content") or f"tarea {i+1}")[:60],
            "inicio": m["timestamp"],
            "fin": fin,
        })
    return tramos


def _variabilidad_expresiva(sub: list[dict]) -> float:
    """
    Cuánto se mueve la cara, sin decir hacia qué emoción.

    Cuenta con qué frecuencia cambia la emoción predominante entre frames
    consecutivos. Una cara quieta da valores cercanos a 0; una muy cambiante,
    cercanos a 1. Que la etiqueta concreta sea poco fiable no invalida la
    medida: lo que se mide es el cambio, no su contenido.
    """
    if len(sub) < 2:
        return 0.0
    cambios = sum(1 for a, b in zip(sub, sub[1:]) if a["emotion"] != b["emotion"])
    return round(cambios / (len(sub) - 1), 3)


def _metricas_tramo(sub: list[dict], inicio: float, fin: float) -> dict:
    duracion = max(fin - inicio, 1e-6)
    n = len(sub)
    if n == 0:
        return {"frames": 0, "duracion_s": round(duracion, 1)}

    sonrisas = [r for r in sub if r["emotion"] == ETIQUETA_FIABLE]
    primera = None
    for r in sub:
        if r["emotion"] == ETIQUETA_FIABLE:
            primera = round(r["timestamp"] - inicio, 1)
            break

    # Un "evento expresivo" es un cambio de emoción sostenido, no cada
    # fluctuación suelta: interesa dónde pasó algo, no el ruido.
    eventos = 0
    anterior = None
    seguidos = 0
    for r in sub:
        if r["emotion"] != anterior:
            if seguidos >= 3:
                eventos += 1
            anterior = r["emotion"]
            seguidos = 1
        else:
            seguidos += 1

    return {
        "frames": n,
        "duracion_s": round(duracion, 1),
        "tiempo_sonriendo_pct": round(len(sonrisas) / n * 100, 1),
        "primera_sonrisa_s": primera,
        "eventos_expresivos": eventos,
        "eventos_por_minuto": round(eventos / (duracion / 60.0), 2),
        "variabilidad_expresiva": _variabilidad_expresiva(sub),
    }


def calcular(registros: list[dict], marcadores: list[dict] | None = None) -> dict:
    """
    Métricas de usabilidad de una sesión.

    Args:
        registros: emotion records ordenados por timestamp, cada uno con
            'timestamp' y 'emotion'.
        marcadores: notas del moderador que delimitan tareas (opcional).

    Returns:
        Métricas globales y por tramo, más los momentos que conviene revisar.
    """
    if not registros:
        return {"disponible": False, "motivo": "la sesión no tiene registros"}

    registros = sorted(registros, key=lambda r: r["timestamp"])
    inicio, fin = registros[0]["timestamp"], registros[-1]["timestamp"]

    global_ = _metricas_tramo(registros, inicio, fin)
    tramos = []
    for t in _tramos(registros, marcadores or []):
        sub = [r for r in registros if t["inicio"] <= r["timestamp"] < t["fin"]]
        tramos.append({**t, **_metricas_tramo(sub, t["inicio"], t["fin"])})

    # Dónde mirar: los tramos más movidos primero. El sistema no interpreta
    # qué pasó, solo ordena la cola de revisión del investigador.
    revisar = sorted(
        [t for t in tramos if t.get("frames")],
        key=lambda t: t.get("variabilidad_expresiva", 0),
        reverse=True,
    )[:5]

    return {
        "disponible": True,
        "global": global_,
        "por_tramo": tramos,
        "revisar_primero": [
            {"tramo": t["nombre"], "inicio_s": round(t["inicio"], 1),
             "variabilidad_expresiva": t.get("variabilidad_expresiva", 0)}
            for t in revisar
        ],
        "nota": (
            "Métricas descriptivas de la sesión, no una evaluación de la persona. "
            "Solo se usan señales que aguantan un cambio de cara: la sonrisa "
            "(95% de acierto sobre 20 actores) y el recuento de cambios "
            "expresivos, que no depende de acertar la emoción."
        ),
    }
