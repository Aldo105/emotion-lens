"""
Banco de pruebas sintetico para HeartRateEstimator.

Alimenta el estimador real con senales rPPG sinteticas de amplitud y ruido
conocidos, sin camara, para medir en cual de sus compuertas se pierde la
lectura. Sirve para reproducir el sintoma "el ritmo cardiaco ya no marca" y
para comprobar que un cambio de parametros lo arregla sin tener que grabar
una sesion nueva cada vez.

Uso:
    python tools/hr_validation/simulate_estimator.py            # las 3 pruebas
    python tools/hr_validation/simulate_estimator.py sensibilidad
    python tools/hr_validation/simulate_estimator.py recuperacion
    python tools/hr_validation/simulate_estimator.py huecos

Requiere numpy y scipy. No requiere opencv: la extraccion de ROI se sustituye
por una senal sintetica, asi que el modulo cv2 se simula si no esta instalado.
"""
import sys
import types
from pathlib import Path

import numpy as np

# El estimador importa cv2 a nivel de modulo, pero aqui nunca se usa: las
# funciones que tocan pixeles se sustituyen abajo. Un stub evita exigir
# opencv solo para correr esta simulacion.
try:
    import cv2  # noqa: F401
except ImportError:
    _stub = types.ModuleType("cv2")
    for _name in ("fillPoly", "GaussianBlur", "polylines", "putText",
                  "rectangle", "getTextSize", "cvtColor", "Laplacian"):
        setattr(_stub, _name, lambda *a, **k: None)
    _stub.LINE_AA = 16
    _stub.FONT_HERSHEY_SIMPLEX = 0
    sys.modules["cv2"] = _stub

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.services.heart_rate import HeartRateEstimator  # noqa: E402

FPS = 20.0          # lo que envia el frontend (config.js, CAPTURE_FPS)
TRUE_BPM = 72.0     # pulso real simulado
LANDMARKS = [{"x": 0.5, "y": 0.5}] * 478
SHAPE = (480, 640)

# Medias de canal de una frente real (0-255) y ganancia relativa del pulso
# por canal: el verde es el que mas modula, segun de Haan & Jeanne (2013).
BASE_RGB = np.array([150.0, 120.0, 110.0])
CHANNEL_GAIN = np.array([0.35, 1.0, 0.65])
NOSE_RGB = np.array([160.0, 130.0, 118.0])


def _build(estimator, amplitude_fn, rng):
    """Sustituye la extraccion de ROI por una senal sintetica."""
    state = {"t": 0.0}
    freq = TRUE_BPM / 60.0

    def forehead(frame, landmarks, frame_shape):
        t = state["t"]
        amp, noise = amplitude_fn(t)
        pulse = amp * np.sin(2 * np.pi * freq * t) * CHANNEL_GAIN
        drift = 2.0 * np.sin(2 * np.pi * 0.05 * t)   # deriva lenta de luz
        v = BASE_RGB + pulse + drift + rng.normal(0, noise, 3)
        return float(v[0]), float(v[1]), float(v[2])

    def nose(frame, landmarks, frame_shape):
        t = state["t"]
        drift = 2.0 * np.sin(2 * np.pi * 0.05 * t)
        v = NOSE_RGB + drift + rng.normal(0, 0.2, 3)
        return float(v[0]), float(v[1]), float(v[2])

    estimator._extract_forehead_signal = forehead
    estimator._extract_nose_bridge_signal = nose
    estimator._detect_motion = lambda landmarks, frame_shape: False
    return state


def sensibilidad(duration=60.0):
    """A que relacion senal/ruido deja de aparecer una lectura."""
    print(f"\nSENSIBILIDAD — pulso real {TRUE_BPM:.0f} BPM, {duration:.0f}s a {FPS:.0f} FPS")
    print("amplitud = modulacion del pulso en niveles de intensidad (8 bits)")
    print("ruido    = desviacion estandar del ruido por frame\n")
    print(f"{'amplitud':>9} {'ruido':>6} {'con lectura':>12} {'1a lectura':>11} "
          f"{'BPM':>7} {'IQR med':>8}")
    print("-" * 60)

    for amp in (1.5, 0.8, 0.4, 0.2):
        for noise in (0.1, 0.3, 0.8):
            rng = np.random.default_rng(1)
            est = HeartRateEstimator(fps=FPS)
            state = _build(est, lambda t: (amp, noise), rng)

            ready = 0
            first = None
            reported = []
            spreads = []
            n = int(duration * FPS)
            for i in range(n):
                t = i / FPS
                state["t"] = t
                r = est.process_frame(None, LANDMARKS, SHAPE, t)
                if r["signal_ready"]:
                    ready += 1
                    reported.append(r["bpm"])
                    if first is None:
                        first = t
                if len(est._raw_history) >= 8:
                    raw = np.array(est._raw_history)
                    spreads.append(np.percentile(raw, 75) - np.percentile(raw, 25))

            eligible = max(n - int(est.min_estimation_seconds * FPS), 1)
            pct = 100.0 * ready / eligible
            first_s = f"{first:.1f}s" if first is not None else "nunca"
            bpm = f"{np.median(reported):.1f}" if reported else "--"
            iqr = f"{np.median(spreads):.1f}" if spreads else "--"
            print(f"{amp:>9.2f} {noise:>6.2f} {pct:>11.1f}% {first_s:>11} {bpm:>7} {iqr:>8}")


def recuperacion(total=90.0):
    """Cuanta senal limpia hace falta para que vuelva tras una mala racha."""
    print(f"\nRECUPERACION — N segundos de senal debil y luego pulso limpio")
    print("'retraso' = segundos de senal LIMPIA hasta que reaparece un valor\n")
    print(f"{'mala racha':>11} {'retraso':>15} {'con lectura':>13}")
    print("-" * 42)

    for bad in (0.0, 5.0, 10.0, 20.0, 30.0):
        rng = np.random.default_rng(3)
        est = HeartRateEstimator(fps=FPS)
        state = _build(est, lambda t: (0.2, 0.9) if t < bad else (1.2, 0.2), rng)

        recovered = None
        ready = after = 0
        for i in range(int(total * FPS)):
            t = i / FPS
            state["t"] = t
            r = est.process_frame(None, LANDMARKS, SHAPE, t)
            if t >= bad:
                after += 1
                if r["signal_ready"]:
                    ready += 1
                    if recovered is None:
                        recovered = t
        lag = f"{recovered - bad:.1f}s" if recovered is not None else "no vuelve"
        print(f"{bad:>10.0f}s {lag:>15} {100.0 * ready / max(after, 1):>12.1f}%")


def huecos(duration=120.0, burst=10):
    """Efecto de perder frames: el pulso es perfecto, solo falta muestreo.

    Reproduce lo que hacen los `continue` de websocket.py (quality_gate ==
    "fail" y rostro no detectado): esos frames nunca llegan al estimador.
    """
    print(f"\nHUECOS DE MUESTREO — pulso PERFECTO de {TRUE_BPM:.0f} BPM, "
          f"{duration:.0f}s")
    print("solo se pierden frames; la senal en si es impecable\n")
    print(f"{'perdida':>8} {'frames vistos':>14} {'con lectura':>12} {'BPM reportado':>14}")
    print("-" * 52)

    for drop in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        rng = np.random.default_rng(7)
        est = HeartRateEstimator(fps=FPS)
        state = _build(est, lambda t: (1.2, 0.2), rng)

        n = int(duration * FPS)
        ready = delivered = 0
        burst_left = 0
        reported = []
        for i in range(n):
            t = i / FPS
            state["t"] = t
            # Perdidas en rafagas: el quality gate se recalcula cada 10 frames,
            # asi que cuando falla arrastra ~10 frames seguidos.
            if burst_left > 0:
                burst_left -= 1
                continue
            if drop > 0 and rng.random() < drop / burst:
                burst_left = burst - 1
                continue
            delivered += 1
            r = est.process_frame(None, LANDMARKS, SHAPE, t)
            if r["signal_ready"]:
                ready += 1
                reported.append(r["bpm"])

        bpm = f"{np.median(reported):.1f}" if reported else "-- (en blanco)"
        print(f"{drop * 100:>7.0f}% {100.0 * delivered / n:>13.1f}% "
              f"{100.0 * ready / max(delivered, 1):>11.1f}% {bpm:>14}")


PRUEBAS = {
    "sensibilidad": sensibilidad,
    "recuperacion": recuperacion,
    "huecos": huecos,
}

if __name__ == "__main__":
    elegidas = sys.argv[1:] or list(PRUEBAS)
    for nombre in elegidas:
        if nombre not in PRUEBAS:
            sys.exit(f"Prueba desconocida: {nombre}. Opciones: {', '.join(PRUEBAS)}")
        PRUEBAS[nombre]()
    print()
