"""
Compara las tres vías de medir expresión sobre los mismos 20 actores.

El proyecto mide la cara por tres caminos distintos:
  1. CNN        — píxeles de la imagen, sin adaptación a la persona
  2. Blendshapes — los 52 coeficientes de MediaPipe, con baseline por sujeto
  3. AUs geométricos — distancias entre landmarks, con baseline por sujeto

Los dos últimos sí se calibran contra el reposo de cada individuo, así que en
teoría deberían aguantar mejor un cambio de cara que el CNN, que cae del 60%
con un actor al 36% con veinte. Esto lo comprueba.

Uso:
    python evaluate_three_paths.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI.parent.parent))

from backend.app.services.emotion_classifier import EmotionClassifier, FER7_LABELS
from backend.app.services.action_units import ActionUnitAnalyzer
from backend.app.services.face_detector import FaceDetector

CLIPS = AQUI / "cremad"
# CREMA-D no tiene 'surprise'; 'contempt'/'stress' no son etiquetas suyas
MAPA_MICRO = {"anger": "angry", "disgust": "disgust", "fear": "fear",
              "sadness": "sad", "surprise": "surprise"}


def analizar_clip(ruta, clf, det, au):
    """Devuelve, por frame, la predicción de cada una de las tres vías."""
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    salida, j = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        j += 1
        if j % 3:
            continue
        d = det.detect(frame, timestamp_ms=int(j / fps * 1000))
        if d is None:
            continue

        fila = {}
        face = EmotionClassifier.preprocess_face(frame, d["bbox"])
        if face is not None:
            p = clf._predict_cnn(face)["probabilities"]
            fila["cnn"] = max((k for k in p if k in FER7_LABELS), key=lambda k: p[k])

        bs = d.get("blendshapes")
        if bs:
            r = clf._predict_blendshapes(clf._subtract_baseline(bs), bs)
            pr = {k: v for k, v in r["probabilities"].items() if k in FER7_LABELS}
            if pr:
                fila["blendshape"] = max(pr, key=pr.get)

        aus = au.compute(landmarks=d["landmarks"],
                         frame_shape=d["frame_shape"], timestamp=j / fps)
        if aus:
            fila["aus"] = aus
        salida.append(fila)
    cap.release()
    return salida


def emocion_por_aus(aus, basal):
    """Puntúa los prototipos EMFACS sobre la desviación respecto al reposo."""
    from backend.app.services.micro_expressions import MICRO_EXPR_PATTERNS
    desv = {k: max(0.0, aus.get(k, 0.0) - basal.get(k, 0.0)) for k in aus}
    mejor, mejor_p = None, 0.0
    for nombre, patron in MICRO_EXPR_PATTERNS.items():
        etiqueta = MAPA_MICRO.get(nombre)
        if not etiqueta:
            continue
        req = [desv.get(a, 0.0) for a in patron["required_aus"]]
        sup = [desv.get(a, 0.0) for a in patron["supporting_aus"]]
        if not req:
            continue
        p = float(np.mean(req)) * 0.7 + (float(np.mean(sup)) * 0.3 if sup else 0.0)
        if p > mejor_p:
            mejor, mejor_p = etiqueta, p
    return mejor if mejor_p > 0.02 else "neutral"


def main():
    indice = CLIPS / "indice.json"
    if not indice.exists():
        sys.exit("Falta cremad/indice.json")
    clips = json.loads(indice.read_text(encoding="utf-8"))["clips"]

    clf = EmotionClassifier()
    if clf.mode != "cnn":
        sys.exit("El CNN no cargo")
    det = FaceDetector(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5)

    por_actor = defaultdict(list)
    for c in clips:
        por_actor[c["actor"]].append(c)

    marcador = {v: defaultdict(lambda: [0, 0]) for v in ("cnn", "blendshape", "aus")}
    por_actor_acierto = {v: {} for v in marcador}

    for actor, lista in sorted(por_actor.items()):
        neutral = next((c for c in lista if c["emocion"] == "neutral"), None)
        if not neutral:
            continue

        # Calibrar con el clip neutro de esta persona
        au = ActionUnitAnalyzer()
        clf._baseline_blendshapes, clf._baseline_set = None, False
        clf._calibration_buffer = []

        base_frames = analizar_clip(CLIPS / neutral["archivo"], clf, det, au)
        aus_basal = {}
        if base_frames:
            claves = {k for f in base_frames if "aus" in f for k in f["aus"]}
            aus_basal = {k: float(np.mean([f["aus"].get(k, 0.0)
                                           for f in base_frames if "aus" in f]))
                         for k in claves}
        cap = cv2.VideoCapture(str(CLIPS / neutral["archivo"]))
        j = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            j += 1
            if j % 3:
                continue
            d = det.detect(frame, timestamp_ms=j * 33)
            if d and d.get("blendshapes"):
                clf.record_calibration_frame(d["blendshapes"])
        cap.release()
        clf.finalize_baseline()

        aciertos = {v: [0, 0] for v in marcador}
        for c in lista:
            if c["emocion"] == "neutral":
                continue
            frames = analizar_clip(CLIPS / c["archivo"], clf, det, au)
            if not frames:
                continue
            real = c["emocion"]

            votos = {
                "cnn": Counter(f["cnn"] for f in frames if "cnn" in f),
                "blendshape": Counter(f["blendshape"] for f in frames if "blendshape" in f),
                "aus": Counter(emocion_por_aus(f["aus"], aus_basal)
                               for f in frames if "aus" in f),
            }
            for via, cnt in votos.items():
                if not cnt:
                    continue
                pred = cnt.most_common(1)[0][0]
                marcador[via][real][1] += 1
                aciertos[via][1] += 1
                if pred == real:
                    marcador[via][real][0] += 1
                    aciertos[via][0] += 1

        for via in marcador:
            if aciertos[via][1]:
                por_actor_acierto[via][actor] = aciertos[via][0] / aciertos[via][1]
        print(f"  actor {actor}: " + "  ".join(
            f"{v}={aciertos[v][0]}/{aciertos[v][1]}" for v in marcador), flush=True)

    det.close()

    print(f"\n{'='*58}")
    print(f"{'via':<14} {'global':>10}   por emocion")
    print("-" * 78)
    for via in ("cnn", "blendshape", "aus"):
        ok = sum(v[0] for v in marcador[via].values())
        n = sum(v[1] for v in marcador[via].values())
        detalle = "  ".join(f"{e[:4]}:{v[0]}/{v[1]}"
                            for e, v in sorted(marcador[via].items()))
        print(f"{via:<14} {ok:>3}/{n:<3} {ok/max(n,1)*100:>3.0f}%   {detalle}")

    print(f"\ndispersion entre actores (min-max):")
    for via in ("cnn", "blendshape", "aus"):
        v = list(por_actor_acierto[via].values())
        if v:
            print(f"  {via:<12} {min(v)*100:>3.0f}% a {max(v)*100:>3.0f}%"
                  f"   (rango {int((max(v)-min(v))*100)} puntos)")
    print("\n  un rango estrecho = funciona parecido con cualquier cara")


if __name__ == "__main__":
    main()
