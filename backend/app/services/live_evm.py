"""
EmotionLens — Magnificación Euleriana (EVM) en vivo

El mismo filtro que produce el video "Generar EVM" de la sección de reportes,
pero aplicado cuadro a cuadro sobre el video en vivo.

Lo que había antes en el toggle no era este filtro: era un tinte plano de color
sobre la frente, cuya intensidad seguía la fase del pulso estimado
(`HeartRateEstimator.get_magnified_frame`). Además dependía de los buffers del
estimador de ritmo cardíaco, así que cuando ese estimador se quedaba sin señal
—que es lo habitual— devolvía el cuadro intacto y el interruptor parecía no
hacer nada. Este módulo no depende del estimador: mantiene su propio estado a
partir de los cuadros que ve.

Pipeline, idéntico al de EVMRenderer salvo el filtro temporal:
  1. Recortar el ROI del rostro y llevarlo a un tamaño de trabajo fijo
  2. BGR -> YIQ
  3. Bajar por una pirámide gaussiana
  4. Filtro pasa-banda temporal en la banda del pulso
  5. Anular el canal Y (magnificación solo de color) y amplificar
  6. Subir por la pirámide, sumar al ROI y fundir con el cuadro original

La diferencia está en el paso 4. El renderer offline tiene el video completo y
usa `filtfilt` sobre todo un bloque de cuadros; en vivo no existe el futuro, así
que se usa el pasa-banda IIR de la formulación en tiempo real del propio
artículo de EVM: la diferencia de dos medias móviles exponenciales, una con
corte en la frecuencia alta de la banda y otra en la baja. Cuesta O(1) por
cuadro y no necesita guardar historial de imágenes.
"""

import cv2
import numpy as np

from backend.app.services.evm_renderer import (
    _bgr_to_yiq,
    _build_pyramid_down,
    _create_feather_mask,
    _upsample,
    _yiq_to_bgr,
)

# El ROI se reescala a este tamaño antes de la pirámide. El recuadro del rostro
# cambia de tamaño con cada cuadro conforme la persona se mueve, y el estado del
# filtro temporal es una imagen de forma fija: sin un tamaño de trabajo estable
# habría que descartarlo en cuanto el encuadre varía un pixel. También acota el
# costo por cuadro, que aquí corre dentro del bucle en vivo.
WORKING_SIZE = 160

# Bajo este tamaño el recorte no da para la pirámide.
MIN_ROI_PX = 24


def _cascade(stages: list[np.ndarray], value: np.ndarray, alpha: float) -> None:
    """Pasa `value` por medias móviles en cascada, actualizándolas in-place."""
    signal = value
    for stage in stages:
        stage += alpha * (signal - stage)
        signal = stage


def _ema_alpha(cutoff_hz: float, fps: float) -> float:
    """Coeficiente de una media móvil exponencial con ese corte."""
    if cutoff_hz <= 0 or fps <= 0:
        return 1.0
    return float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz / fps))


class LiveEVMMagnifier:
    """
    Aplica magnificación de color Euleriana al rostro, cuadro a cuadro.

    El estado son dos imágenes del tamaño del nivel de pirámide (unos pocos
    cientos de floats), así que una instancia por sesión no pesa.
    """

    def __init__(
        self,
        amplification: float = 40.0,
        freq_low: float = 0.7,
        freq_high: float = 3.0,
        pyramid_levels: int = 4,
        fps: float = 20.0,
    ):
        self.amplification = amplification
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.pyramid_levels = pyramid_levels
        self.fps = fps

        self._alpha_high = _ema_alpha(freq_high, fps)
        self._alpha_low = _ema_alpha(freq_low, fps)

        # Dos secciones en cascada por corte en vez de una. Con una sola, la
        # caída fuera de banda es de 6 dB/octava: en las pruebas, una
        # oscilación a 6 Hz —ruido de sensor o movimiento, no pulso— salía
        # amplificada casi tanto como el pulso, y ×40 eso se ve como una cara
        # llena de grano. La cascada duplica la pendiente y acerca el resultado
        # al Butterworth de orden 3 que usa el render offline, sin dejar de
        # costar O(1) por cuadro.
        self._high: list[np.ndarray] | None = None
        self._low: list[np.ndarray] | None = None
        self._feather: np.ndarray | None = None
        self._frames_seen = 0

    def reset(self) -> None:
        """Olvida el estado temporal (p. ej. al recalibrar)."""
        self._high = None
        self._low = None
        self._frames_seen = 0

    @property
    def warmed_up(self) -> bool:
        """
        Si el filtro ya arrancó.

        Las dos medias móviles empiezan iguales, así que su diferencia nace en
        cero y tarda en separarse. Amplificar ×40 ese transitorio pinta un
        destello de color que no es pulso, de modo que los primeros cuadros
        salen sin amplificar.
        """
        return self._frames_seen > self.fps  # ~1 s

    def process(self, frame: np.ndarray, bbox: dict | None) -> np.ndarray:
        """
        Devuelve el cuadro con el rostro magnificado.

        Si no hay rostro o el recorte es demasiado pequeño, devuelve el cuadro
        tal cual: sin ROI estable el filtro compararía regiones distintas entre
        cuadros y amplificaría ese desajuste, no el flujo sanguíneo.
        """
        if bbox is None:
            return frame

        h, w = frame.shape[:2]
        x1 = max(0, int(bbox["x_min"]))
        y1 = max(0, int(bbox["y_min"]))
        x2 = min(w, int(bbox["x_max"]))
        y2 = min(h, int(bbox["y_max"]))

        roi_h, roi_w = y2 - y1, x2 - x1
        if roi_h < MIN_ROI_PX or roi_w < MIN_ROI_PX:
            return frame

        roi = frame[y1:y2, x1:x2]
        working = cv2.resize(roi, (WORKING_SIZE, WORKING_SIZE), interpolation=cv2.INTER_AREA)
        working_yiq = _bgr_to_yiq(working.astype(np.float32) / 255.0)
        level = _build_pyramid_down(working_yiq, self.pyramid_levels)

        bandpassed = self._advance(level)
        if bandpassed is None:
            return frame

        # Solo color: el canal Y lleva la luminancia, y amplificarla resalta el
        # ruido del sensor y el movimiento en vez del cambio de color de la
        # piel. El renderer offline hace lo mismo.
        bandpassed = bandpassed.copy()
        bandpassed[:, :, 0] = 0.0
        bandpassed *= self.amplification

        upsampled = _upsample(bandpassed, self.pyramid_levels, WORKING_SIZE, WORKING_SIZE)
        magnified = _yiq_to_bgr(working_yiq + upsampled)
        magnified = np.clip(magnified * 255.0, 0, 255).astype(np.uint8)

        magnified_roi = cv2.resize(magnified, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        feather = self._feather_for(roi_h, roi_w)
        blended = (
            magnified_roi.astype(np.float32) * feather
            + roi.astype(np.float32) * (1.0 - feather)
        )

        output = frame.copy()
        output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        _draw_badge(output)
        return output

    def _advance(self, level: np.ndarray) -> np.ndarray | None:
        """Actualiza las medias móviles y devuelve la banda, o None si aún no."""
        if self._high is None or self._high[0].shape != level.shape:
            # Primer cuadro, o cambió la forma del nivel: arrancar de cero en
            # vez de restar imágenes de tamaños distintos.
            self._high = [level.copy(), level.copy()]
            self._low = [level.copy(), level.copy()]
            self._frames_seen = 1
            return None

        _cascade(self._high, level, self._alpha_high)
        _cascade(self._low, level, self._alpha_low)
        self._frames_seen += 1

        if not self.warmed_up:
            return None

        return self._high[-1] - self._low[-1]

    def _feather_for(self, roi_h: int, roi_w: int) -> np.ndarray:
        if self._feather is None or self._feather.shape[:2] != (roi_h, roi_w):
            border = max(4, min(15, roi_h // 8, roi_w // 8))
            self._feather = _create_feather_mask(roi_h, roi_w, border=border)
        return self._feather


def _draw_badge(frame: np.ndarray) -> None:
    """Marca 'EVM' en la esquina, igual que el video del reporte."""
    text = "EVM"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
    x = frame.shape[1] - tw - 14
    cv2.rectangle(frame, (x - 6, 8), (x + tw + 6, th + 18), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, th + 12), font, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
