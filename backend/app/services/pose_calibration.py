"""
EmotionLens — Calibración guiada por pose de cabeza

La calibración original promediaba 30 s de rostro neutro mirando al frente y
usaba esa única media como línea base para todo lo demás. Las Action Units se
calculan sobre distancias entre landmarks 2D, y girar la cabeza acorta esas
distancias por escorzo: con una sola línea base frontal, cualquier frame con la
cabeza girada aparece como una desviación respecto al reposo, aunque la cara no
haya cambiado de expresión. Eso infla el detector de microexpresiones y sesga
al clasificador justo cuando la persona deja de mirar a la cámara, que en una
entrevista real es la mayor parte del tiempo.

Este módulo calibra cinco poses (centro, izquierda, derecha, arriba, abajo) y
usa la diferencia entre ellas para llevar cualquier frame al marco de
referencia frontal antes de que el resto del pipeline lo vea. Los consumidores
(micro_expressions, congruence, emotion_classifier) siguen trabajando contra la
línea base del centro y no necesitan cambios.

El ángulo de cabeza viene de _estimate_head_pose (websocket.py), en grados
aproximados: yaw positivo = la nariz se desplaza hacia la derecha del encuadre,
pitch positivo = la nariz baja respecto al eje frente-mentón.
"""

from dataclasses import dataclass, field

import numpy as np


# Anclas de cada pose en (yaw, pitch) grados, en la escala de
# _estimate_head_pose. Son objetivos de captura, no medidas de un sujeto real:
# se piden giros cómodos, no extremos, porque MediaPipe pierde landmarks más
# allá de ~35 grados y la línea base dejaría de ser fiable.
POSE_ANCHORS: dict[str, tuple[float, float]] = {
    "center": (0.0, 0.0),
    "right": (22.0, 0.0),
    "left": (-22.0, 0.0),
    "down": (0.0, 18.0),
    "up": (0.0, -18.0),
}

# Orden de la secuencia guiada. El centro va primero: es la pose que alimenta
# la línea base tradicional (umbrales de microexpresión, congruencia), así que
# conviene capturarla mientras la persona aún está quieta y atenta.
POSE_SEQUENCE: tuple[str, ...] = ("center", "right", "left", "down", "up")

# Etiquetas para la instrucción en pantalla.
POSE_LABELS: dict[str, str] = {
    "center": "Mira al frente, rostro relajado",
    "right": "Gira la cabeza a tu derecha",
    "left": "Gira la cabeza a tu izquierda",
    "down": "Baja la barbilla, mira hacia abajo",
    "up": "Sube la barbilla, mira hacia arriba",
}

# Cuánto puede alejarse la pose real del ancla para que la muestra cuente.
# Amplio a propósito: el objetivo es cubrir una región de ángulos, no clavar un
# valor exacto, y un margen estrecho deja a la persona atrapada intentando
# acertar un número que no ve.
POSE_TOLERANCE_DEG: float = 12.0

# El centro es la excepción: define el reposo frontal, así que se exige que
# esté realmente de frente.
CENTER_TOLERANCE_DEG: float = 8.0

# Exponente de la ponderación por distancia inversa (Shepard). Se usa un
# interpolador y no un promedio ponderado con núcleo gaussiano: el promedio no
# reproduce el valor del ancla cuando se consulta justo en el ancla, así que
# corregía frames frontales contra una línea base contaminada por las poses
# giradas. Shepard sí pasa exactamente por cada ancla, y entre ellas varía de
# forma continua, que es lo que evita el salto de línea base que aguas abajo se
# detectaría como una microexpresión falsa.
INTERPOLATION_POWER: float = 2.0

# Distancia (grados) por debajo de la cual se considera que la consulta ES el
# ancla, para no dividir entre cero.
ANCHOR_EPSILON_DEG: float = 1e-6

# Alcance de la corrección. Dentro de CORRECTION_FULL_DEG del ancla más cercana
# la corrección se aplica entera; más allá de CORRECTION_ZERO_DEG no se aplica
# nada. Fuera de la región calibrada no hay datos que respalden una
# extrapolación, y una corrección grande e inventada hace más daño que no
# corregir: el frame se deja tal cual y se trata como frontal.
CORRECTION_FULL_DEG: float = 18.0
CORRECTION_ZERO_DEG: float = 40.0


def pose_tolerance(pose: str) -> float:
    return CENTER_TOLERANCE_DEG if pose == "center" else POSE_TOLERANCE_DEG


def pose_matches(pose: str, yaw: float, pitch: float) -> bool:
    """Si la pose medida está lo bastante cerca del ancla pedida."""
    target_yaw, target_pitch = POSE_ANCHORS[pose]
    distance = np.hypot(yaw - target_yaw, pitch - target_pitch)
    return bool(distance <= pose_tolerance(pose))


@dataclass
class PoseProgress:
    """Estado de la secuencia, para la instrucción en pantalla."""
    pose: str
    label: str
    pose_index: int
    pose_total: int
    samples: int
    samples_required: int
    aligned: bool
    complete: bool

    @property
    def pose_progress(self) -> float:
        return min(1.0, self.samples / max(self.samples_required, 1))

    @property
    def total_progress(self) -> float:
        done = self.pose_index + self.pose_progress
        return min(1.0, done / max(self.pose_total, 1))

    def as_dict(self) -> dict:
        return {
            "pose": self.pose,
            "label": self.label,
            "pose_index": self.pose_index,
            "pose_total": self.pose_total,
            "samples": self.samples,
            "samples_required": self.samples_required,
            "aligned": self.aligned,
            "complete": self.complete,
            "pose_progress": round(self.pose_progress, 3),
            "total_progress": round(self.total_progress, 3),
        }


class PoseCalibrationSession:
    """
    Conduce la captura guiada: indica qué pose sostener, acepta muestras solo
    cuando el ángulo medido coincide, y avanza cuando hay suficientes.

    Se avanza por número de muestras válidas y no por tiempo: una pose que la
    persona no logra sostener no debe darse por calibrada solo porque pasaron
    los segundos. `timed_out` existe para que el llamador decida qué hacer si
    alguien no consigue una pose.
    """

    def __init__(
        self,
        samples_per_pose: int | dict[str, int] = 120,
        seconds_per_pose_limit: float = 25.0,
        sequence: tuple[str, ...] = POSE_SEQUENCE,
    ):
        # El centro admite un conteo propio: además de su línea base, es el que
        # alimenta la variabilidad por AU con la que el motor de
        # microexpresiones fija sus umbrales, y esa estimación necesita más
        # muestras que una simple media.
        if isinstance(samples_per_pose, dict):
            self._required = {p: samples_per_pose.get(p, 120) for p in sequence}
        else:
            self._required = {p: samples_per_pose for p in sequence}
        self.seconds_per_pose_limit = seconds_per_pose_limit
        self.sequence = sequence

        self._index = 0
        self._samples: dict[str, list[dict[str, float]]] = {p: [] for p in sequence}
        self._blendshapes: dict[str, list[dict[str, float]]] = {p: [] for p in sequence}
        self._pose_started_at: float | None = None
        self._complete = False

    @property
    def current_pose(self) -> str | None:
        if self._complete or self._index >= len(self.sequence):
            return None
        return self.sequence[self._index]

    @property
    def complete(self) -> bool:
        return self._complete

    def timed_out(self, timestamp: float) -> bool:
        """Si la pose actual lleva demasiado tiempo sin completarse."""
        if self._complete or self._pose_started_at is None:
            return False
        return (timestamp - self._pose_started_at) > self.seconds_per_pose_limit

    def record(
        self,
        action_units: dict[str, float],
        blendshapes: dict[str, float] | None,
        yaw: float,
        pitch: float,
        timestamp: float,
    ) -> PoseProgress:
        """Ofrece un frame a la calibración y devuelve el estado actualizado."""
        pose = self.current_pose
        if pose is None:
            return self._progress(self.sequence[-1], aligned=False)

        if self._pose_started_at is None:
            self._pose_started_at = timestamp

        aligned = pose_matches(pose, yaw, pitch)
        if aligned:
            self._samples[pose].append(dict(action_units))
            if blendshapes:
                self._blendshapes[pose].append(dict(blendshapes))

            if len(self._samples[pose]) >= self._required[pose]:
                self._advance()

        # Reports the pose now being asked for, which after an advance is the
        # next one: the caller renders this straight to the screen, and naming
        # the pose just finished would flash a stale instruction for a frame.
        return self._progress(self.current_pose or self.sequence[-1], aligned=aligned)

    def progress(self) -> PoseProgress:
        """Estado actual sin ofrecer ninguna muestra."""
        pose = self.current_pose or self.sequence[-1]
        return self._progress(pose, aligned=False)

    def skip_current_pose(self) -> None:
        """Abandona la pose actual y pasa a la siguiente."""
        if not self._complete:
            self._advance()

    def _advance(self) -> None:
        self._index += 1
        self._pose_started_at = None
        if self._index >= len(self.sequence):
            self._complete = True

    def _progress(self, pose: str, aligned: bool) -> PoseProgress:
        return PoseProgress(
            pose=pose,
            label=POSE_LABELS.get(pose, pose),
            pose_index=min(self._index, len(self.sequence) - 1),
            pose_total=len(self.sequence),
            samples=len(self._samples[pose]),
            samples_required=self._required[pose],
            aligned=aligned,
            complete=self._complete,
        )

    def center_samples(self) -> list[dict[str, float]]:
        """Muestras frontales, que alimentan la línea base tradicional."""
        return self._samples.get("center", [])

    def center_blendshapes(self) -> list[dict[str, float]]:
        return self._blendshapes.get("center", [])

    def build_baselines(self) -> "PoseBaselines":
        """Promedia cada pose capturada en su línea base."""
        au_means = {
            pose: _mean_of_dicts(samples)
            for pose, samples in self._samples.items()
            if samples
        }
        blend_means = {
            pose: _mean_of_dicts(samples)
            for pose, samples in self._blendshapes.items()
            if samples
        }
        return PoseBaselines(action_units=au_means, blendshapes=blend_means)


def _mean_of_dicts(samples: list[dict[str, float]]) -> dict[str, float]:
    keys: set[str] = set()
    for sample in samples:
        keys.update(sample.keys())
    return {
        key: float(np.mean([s[key] for s in samples if key in s]))
        for key in keys
    }


@dataclass
class PoseBaselines:
    """
    Líneas base por pose, y la corrección que lleva un frame al marco frontal.

    `to_frontal` resta la diferencia entre la línea base interpolada para el
    ángulo actual y la línea base del centro. Lo que queda es la desviación
    atribuible a la expresión y no a la geometría de la cabeza, que es lo que
    el resto del pipeline siempre asumió estar recibiendo.
    """

    action_units: dict[str, dict[str, float]] = field(default_factory=dict)
    blendshapes: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def calibrated_poses(self) -> list[str]:
        return sorted(self.action_units.keys())

    @property
    def usable(self) -> bool:
        """La corrección necesita el centro y al menos una pose girada."""
        return "center" in self.action_units and len(self.action_units) >= 2

    def _distances(self, yaw: float, pitch: float) -> dict[str, float]:
        return {
            pose: float(np.hypot(yaw - POSE_ANCHORS[pose][0], pitch - POSE_ANCHORS[pose][1]))
            for pose in self.calibrated_poses
        }

    def weights(self, yaw: float, pitch: float) -> dict[str, float]:
        """Peso de cada pose calibrada para un ángulo dado (suman 1)."""
        distances = self._distances(yaw, pitch)
        if not distances:
            return {}

        for pose, distance in distances.items():
            if distance < ANCHOR_EPSILON_DEG:
                return {p: 1.0 if p == pose else 0.0 for p in distances}

        raw = {p: d ** -INTERPOLATION_POWER for p, d in distances.items()}
        total = sum(raw.values())
        return {pose: value / total for pose, value in raw.items()}

    def correction_strength(self, yaw: float, pitch: float) -> float:
        """Cuánto de la corrección aplicar, según lo lejos que caiga del ancla."""
        distances = self._distances(yaw, pitch)
        if not distances:
            return 0.0

        nearest = min(distances.values())
        if nearest <= CORRECTION_FULL_DEG:
            return 1.0
        if nearest >= CORRECTION_ZERO_DEG:
            return 0.0
        span = CORRECTION_ZERO_DEG - CORRECTION_FULL_DEG
        return float(1.0 - (nearest - CORRECTION_FULL_DEG) / span)

    def _interpolated(
        self,
        table: dict[str, dict[str, float]],
        yaw: float,
        pitch: float,
    ) -> dict[str, float]:
        weights = {p: w for p, w in self.weights(yaw, pitch).items() if p in table}
        total = sum(weights.values())
        if total <= 0:
            return {}

        blended: dict[str, float] = {}
        for pose, weight in weights.items():
            share = weight / total
            for key, value in table[pose].items():
                blended[key] = blended.get(key, 0.0) + share * value
        return blended

    def to_frontal(
        self,
        action_units: dict[str, float],
        yaw: float,
        pitch: float,
    ) -> dict[str, float]:
        """Quita de las AUs el componente que aporta el giro de cabeza."""
        return self._correct(self.action_units, action_units, yaw, pitch)

    def blendshapes_to_frontal(
        self,
        blendshapes: dict[str, float],
        yaw: float,
        pitch: float,
    ) -> dict[str, float]:
        return self._correct(self.blendshapes, blendshapes, yaw, pitch)

    def _correct(
        self,
        table: dict[str, dict[str, float]],
        values: dict[str, float],
        yaw: float,
        pitch: float,
    ) -> dict[str, float]:
        if not values or "center" not in table or len(table) < 2:
            return values

        strength = self.correction_strength(yaw, pitch)
        if strength <= 0.0:
            return values

        here = self._interpolated(table, yaw, pitch)
        center = table["center"]
        if not here:
            return values

        corrected = {}
        for key, value in values.items():
            offset = here.get(key, center.get(key, 0.0)) - center.get(key, 0.0)
            corrected[key] = value - strength * offset
        return corrected
