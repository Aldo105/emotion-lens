"""
¿Cuánto del 36% en CREMA-D es resolución y cuánto es cambiar de cara?

CREMA-D es de 480x360 y RAVDESS de 1280x720, así que la caída entre ambos
mezcla dos causas. Este script evalúa los mismos clips de RAVDESS a su
resolución original y reducida a la de CREMA-D: lo que cambie es resolución,
lo que quede es generalización entre personas.

Para que la comparación valga, se usan solo las 6 emociones que CREMA-D
también tiene (sin 'surprise' ni 'calm') y la misma métrica: clasificador
directo, emoción predominante por clip, sin baseline por sujeto.

Uso:
    python evaluate_resolution.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI.parent.parent))

from backend.app.services.emotion_classifier import EmotionClassifier, FER7_LABELS
from backend.app.services.face_detector import FaceDetector

# Las mismas 6 clases que CREMA-D
EMO = {"01": "neutral", "03": "happy", "04": "sad",
       "05": "angry", "06": "fear", "07": "disgust"}
RES_CREMAD = (480, 360)


def clips_ravdess():
    out = []
    for p in sorted(AQUI.rglob("*.mp4")):
        if "cremad" in p.parts:
            continue
        q = p.stem.split("-")
        if len(q) >= 7 and q[0] == "01" and q[2] in EMO:
            out.append((p, EMO[q[2]], "alta" if q[3] == "02" else "baja"))
    return out


def evaluar(clips, clf, det, escalar_a=None):
    por_emocion = defaultdict(lambda: [0, 0])
    confusion = defaultdict(Counter)
    for ruta, real, _inten in clips:
        cap = cv2.VideoCapture(str(ruta))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        preds, j = Counter(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            j += 1
            if j % 3:
                continue
            if escalar_a:
                # Reducir y volver a ampliar: así se pierde detalle de verdad,
                # que es lo que distingue una webcam mala de uno buena.
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, escalar_a, interpolation=cv2.INTER_AREA)
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            d = det.detect(frame, timestamp_ms=int(j / fps * 1000))
            if d is None:
                continue
            face = EmotionClassifier.preprocess_face(frame, d["bbox"])
            if face is None:
                continue
            p = clf._predict_cnn(face)["probabilities"]
            preds[max((k for k in p if k in FER7_LABELS), key=lambda k: p[k])] += 1
        cap.release()
        if not preds:
            continue
        pred = preds.most_common(1)[0][0]
        confusion[real][pred] += 1
        por_emocion[real][1] += 1
        if pred == real:
            por_emocion[real][0] += 1
    return por_emocion, confusion


def main():
    clips = clips_ravdess()
    if not clips:
        sys.exit("No encontre clips de RAVDESS")
    print(f"{len(clips)} clips de RAVDESS (1 actor), 6 emociones comunes con CREMA-D\n")

    clf = EmotionClassifier()
    if clf.mode != "cnn":
        sys.exit("El CNN no cargo")
    det = FaceDetector(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5)

    resultados = {}
    for etiqueta, escala in (("original 1280x720", None),
                             (f"reducido a {RES_CREMAD[0]}x{RES_CREMAD[1]}", RES_CREMAD)):
        print(f"evaluando {etiqueta}...", flush=True)
        resultados[etiqueta] = evaluar(clips, clf, det, escala)

    det.close()

    print(f"\n{'emocion':<10} " + " ".join(f"{k:>22}" for k in resultados))
    print("-" * 60)
    emociones = sorted({e for r, _ in resultados.values() for e in r})
    for emo in emociones:
        fila = f"{emo:<10} "
        for k in resultados:
            ok, n = resultados[k][0][emo]
            fila += f"{ok:>3}/{n:<3} {ok/max(n,1)*100:>3.0f}%".rjust(23)
        print(fila)
    print("-" * 60)
    fila = f"{'TOTAL':<10} "
    totales = {}
    for k in resultados:
        ok = sum(v[0] for v in resultados[k][0].values())
        n = sum(v[1] for v in resultados[k][0].values())
        totales[k] = ok / max(n, 1) * 100
        fila += f"{ok:>3}/{n:<3} {totales[k]:>3.0f}%".rjust(23)
    print(fila)

    valores = list(totales.values())
    caida = valores[0] - valores[1]
    print(f"\ncaida por resolucion: {caida:+.0f} puntos")
    print(f"CREMA-D (20 actores, 480x360): 36%")
    print(f"\ninterpretacion:")
    if abs(caida) < 5:
        print("  la resolucion no explica la caida -> es generalizacion entre caras")
    elif caida >= valores[0] - 40:
        print("  buena parte de la caida es resolucion, no solo cambiar de persona")
    else:
        print("  la resolucion pesa algo, pero no alcanza a explicar la diferencia")


if __name__ == "__main__":
    main()
