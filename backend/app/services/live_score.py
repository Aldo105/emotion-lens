"""
Índice de desempeño en vivo.

El reporte final ya publica una puntuación general, pero se calcula al cerrar
la sesión: necesita el historial completo (tasa de transiciones, picos de
nerviosismo, declive de confianza). Durante la entrevista no había ningún
número equivalente, así que el panel en vivo no decía si la cosa iba bien.

Este módulo produce un número corriente a partir de lo que sí existe frame a
frame: la valencia de la distribución de emociones y la congruencia. No es la
misma métrica que `interview_analyzer.overall_score` ni pretende predecirla —
esa pondera cinco dimensiones que aquí no se pueden medir todavía.

Todas las constantes son HEURÍSTICAS y están registradas como tales en
`references.py`.
"""

from collections import deque
from dataclasses import dataclass


# Valencia por emoción, en [-1, +1]. Positiva sube el índice, negativa lo baja.
EMOTION_VALENCE: dict[str, float] = {
    "happy": 1.0,
    "confidence": 0.8,
    "surprise": 0.2,
    "neutral": 0.0,
    "nervousness": -0.5,
    "disgust": -0.7,
    "sad": -0.8,
    "angry": -1.0,
}

# Peso de cada componente en el índice.
VALENCE_WEIGHT = 0.6
CONGRUENCE_WEIGHT = 0.4

# Ventana de promediado. Se promedia por tiempo y no por número de frames
# porque la tasa de entrega del navegador es irregular: un promedio sobre las
# últimas N muestras cubre más o menos segundos según cómo vaya el pipeline.
SMOOTHING_SECONDS = 3.0

# Antes de esto el promedio se apoya en muy pocas muestras y salta de golpe;
# se prefiere no mostrar número a mostrar uno que cambia solo por ruido.
MIN_SECONDS_BEFORE_REPORTING = 2.0


@dataclass
class LiveScoreResult:
    score: float | None      # 0-100, o None mientras no hay suficiente señal
    valence: float           # -1..+1, componente emocional crudo
    congruence: float | None # 0-100, tal como entró
    ready: bool
    samples: int

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 1) if self.score is not None else None,
            "valence": round(self.valence, 3),
            "congruence": round(self.congruence, 1) if self.congruence is not None else None,
            "ready": self.ready,
            "samples": self.samples,
        }


def emotion_valence(probabilities: dict[str, float]) -> float:
    """
    Valencia de la distribución completa, no solo de la emoción dominante: una
    predicción repartida entre 'happy' y 'sad' describe algo distinto a una
    'happy' rotunda, y colapsarla al argmax pierde justo esa diferencia.
    """
    if not probabilities:
        return 0.0

    total = sum(probabilities.values())
    if total <= 0:
        return 0.0

    valence = sum(
        prob * EMOTION_VALENCE.get(label, 0.0)
        for label, prob in probabilities.items()
    )
    return max(-1.0, min(1.0, valence / total))


class LiveScoreTracker:
    """Mantiene el índice promediado sobre una ventana temporal."""

    def __init__(
        self,
        smoothing_seconds: float = SMOOTHING_SECONDS,
        min_seconds: float = MIN_SECONDS_BEFORE_REPORTING,
    ):
        self.smoothing_seconds = smoothing_seconds
        self.min_seconds = min_seconds
        self._samples: deque[tuple[float, float]] = deque()
        self._first_timestamp: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._first_timestamp = None

    def update(
        self,
        probabilities: dict[str, float],
        congruence: float | None,
        timestamp: float,
    ) -> LiveScoreResult:
        valence = emotion_valence(probabilities)
        valence_score = (valence + 1.0) * 50.0

        if congruence is None:
            raw = valence_score
        else:
            congruence = max(0.0, min(100.0, congruence))
            raw = (valence_score * VALENCE_WEIGHT) + (congruence * CONGRUENCE_WEIGHT)

        if self._first_timestamp is None:
            self._first_timestamp = timestamp

        self._samples.append((timestamp, raw))
        cutoff = timestamp - self.smoothing_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        elapsed = timestamp - self._first_timestamp
        ready = elapsed >= self.min_seconds and len(self._samples) > 1

        score = None
        if ready:
            score = sum(value for _, value in self._samples) / len(self._samples)
            score = max(0.0, min(100.0, score))

        return LiveScoreResult(
            score=score,
            valence=valence,
            congruence=congruence,
            ready=ready,
            samples=len(self._samples),
        )
