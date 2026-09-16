"""
¿Arreglaría la generalización normalizar las salidas del CNN por persona?

El sistema calibra 30s con la cara del sujeto, pero ese baseline solo se aplica
a los blendshapes y a las etiquetas derivadas: el CNN recibe la imagen cruda.
Nada compensa que cada cara en reposo sea distinta, que es la explicación más
probable del 36% sobre 20 actores frente al ~60% sobre uno.

Aquí se prueba la idea equivalente a lo que ya se hace con blendshapes: medir
el perfil de probabilidades que el CNN da sobre la cara NEUTRA de cada actor y
descontarlo de sus demás clips. Si una cara tira de por sí hacia 'sad', eso lo
corrige.

Uso:
    python evaluate_baseline_cnn.py
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
from backend.app.services.face_detector import FaceDetector

CLIPS = AQUI / "cremad"


def probs_de_clip(ruta, clf, det):
    """Lista de distribuciones de probabilidad, una por frame analizado."""
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, j = [], 0
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
        face = EmotionClassifier.preprocess_face(frame, d["bbox"])
        if face is None:
            continue
        out.append(clf._predict_cnn(face)["probabilities"])
    cap.release()
    return out


def predominante(lista_probs, basal=None):
    """Emoción más frecuente; si hay basal, se descuenta antes."""
    votos = Counter()
    for p in lista_probs:
        if basal:
            # Descontar el sesgo propio de esa cara y renormalizar
            ajust = {k: max(0.0, p.get(k, 0.0) - basal.get(k, 0.0)) for k in FER7_LABELS}
            total = sum(ajust.values())
            if total <= 1e-6:
                continue
            p = {k: v / total for k, v in ajust.items()}
        votos[max((k for k in p if k in FER7_LABELS), key=lambda k: p[k])] += 1
    return votos.most_common(1)[0][0] if votos else None


def main():
    indice = CLIPS / "indice.json"
    if not indice.exists():
        sys.exit("Falta cremad/indice.json — corre download_cremad.py")
    clips = json.loads(indice.read_text(encoding="utf-8"))["clips"]

    clf = EmotionClassifier()
    if clf.mode != "cnn":
        sys.exit("El CNN no cargo")
    det = FaceDetector(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5)

    por_actor = defaultdict(list)
    for c in clips:
        por_actor[c["actor"]].append(c)

    sin_norm = defaultdict(lambda: [0, 0])
    con_norm = defaultdict(lambda: [0, 0])
    actor_sin, actor_con = {}, {}

    for n_actor, (actor, lista) in enumerate(sorted(por_actor.items()), 1):
        neutral = next((c for c in lista if c["emocion"] == "neutral"), None)
        if not neutral:
            continue

        # Perfil basal: lo que el CNN ve en la cara neutra de esta persona
        probs_neutral = probs_de_clip(CLIPS / neutral["archivo"], clf, det)
        if not probs_neutral:
            continue
        basal = {k: float(np.mean([p.get(k, 0.0) for p in probs_neutral]))
                 for k in FER7_LABELS}

        a_sin = a_con = total = 0
        for c in lista:
            if c["emocion"] == "neutral":
                continue          # se usó para calibrar
            probs = probs_de_clip(CLIPS / c["archivo"], clf, det)
            if not probs:
                continue
            real = c["emocion"]
            total += 1
            p1, p2 = predominante(probs), predominante(probs, basal)
            sin_norm[real][1] += 1
            con_norm[real][1] += 1
            if p1 == real:
                sin_norm[real][0] += 1
                a_sin += 1
            if p2 == real:
                con_norm[real][0] += 1
                a_con += 1

        if total:
            actor_sin[actor] = a_sin / total
            actor_con[actor] = a_con / total
        print(f"  actor {actor}: sin {a_sin}/{total}  con {a_con}/{total}", flush=True)

    det.close()

    def resumen(d):
        ok = sum(v[0] for v in d.values())
        n = sum(v[1] for v in d.values())
        return ok, n, ok / max(n, 1) * 100

    ok1, n1, pct1 = resumen(sin_norm)
    ok2, n2, pct2 = resumen(con_norm)

    print(f"\n{'='*54}")
    print(f"SIN normalizar por persona : {ok1}/{n1} ({pct1:.0f}%)")
    print(f"CON normalizar por persona : {ok2}/{n2} ({pct2:.0f}%)")
    print(f"diferencia: {pct2-pct1:+.0f} puntos")

    print(f"\n{'emocion':<10} {'sin':>10} {'con':>10}")
    for emo in sorted(sin_norm):
        a, n = sin_norm[emo]
        b, _ = con_norm[emo]
        print(f"  {emo:<8} {a:>3}/{n:<3} {a/n*100:>3.0f}% {b:>3}/{n:<3} {b/n*100:>3.0f}%")

    if actor_sin:
        v1, v2 = list(actor_sin.values()), list(actor_con.values())
        print(f"\ndispersion entre actores:")
        print(f"  sin normalizar: {min(v1)*100:.0f}% a {max(v1)*100:.0f}%")
        print(f"  con normalizar: {min(v2)*100:.0f}% a {max(v2)*100:.0f}%")
        print("  (si el rango se estrecha, la normalizacion iguala entre personas)")


if __name__ == "__main__":
    main()
