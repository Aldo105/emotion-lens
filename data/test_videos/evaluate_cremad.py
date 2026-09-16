"""
Evalúa el clasificador sobre los 220 clips de CREMA-D (20 actores).

Responde dos preguntas que con un solo actor no se podían contestar:
  1. ¿Lo ajustado hoy generaliza a otras caras?
  2. ¿Funciona igual para todos los grupos demográficos?

Mide el clasificador directamente, no el sistema completo: los clips duran
2-3s y provienen de 20 personas distintas, así que no hay forma de calibrar
un baseline por sujeto como en una sesión real. Lo que cambió hoy (escala de
grises, recorte) actúa en esta etapa, que es justo la que interesa comprobar.

Uso:
    python evaluate_cremad.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI.parent.parent))

from backend.app.services.emotion_classifier import EmotionClassifier, FER7_LABELS
from backend.app.services.face_detector import FaceDetector

CLIPS = AQUI / "cremad"


def tabla(titulo, datos, minimo=1):
    """datos: {clave: [aciertos, total]}"""
    print(f"\n{titulo}")
    for clave in sorted(datos, key=lambda k: -datos[k][1]):
        ok, n = datos[clave]
        if n < minimo:
            continue
        barra = "#" * int(ok / max(n, 1) * 20)
        print(f"  {str(clave):<20} {ok:>3}/{n:<3} {ok/max(n,1)*100:>3.0f}%  {barra}")


def main():
    indice_path = CLIPS / "indice.json"
    if not indice_path.exists():
        sys.exit("Falta cremad/indice.json — corre download_cremad.py")

    clips = json.loads(indice_path.read_text(encoding="utf-8"))["clips"]
    print(f"{len(clips)} clips de {len(set(c['actor'] for c in clips))} actores\n")

    clf = EmotionClassifier()
    print(f"modo: {clf.mode}")
    if clf.mode != "cnn":
        print("AVISO: el CNN no cargó; se mide el camino de respaldo")
    det = FaceDetector(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5)

    por_emocion = defaultdict(lambda: [0, 0])
    por_intensidad = defaultdict(lambda: [0, 0])
    por_sexo = defaultdict(lambda: [0, 0])
    por_etnia = defaultdict(lambda: [0, 0])
    por_edad = defaultdict(lambda: [0, 0])
    por_actor = defaultdict(lambda: [0, 0])
    confusion = defaultdict(Counter)
    sin_cara = 0

    for i, c in enumerate(clips, 1):
        ruta = CLIPS / c["archivo"]
        if not ruta.exists():
            continue

        cap = cv2.VideoCapture(str(ruta))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        preds, j = Counter(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            j += 1
            if j % 3:                      # 1 de cada 3 frames
                continue
            d = det.detect(frame, timestamp_ms=int(j / fps * 1000))
            if d is None:
                continue
            face = EmotionClassifier.preprocess_face(frame, d["bbox"])
            if face is None:
                continue
            p = clf._predict_cnn(face)["probabilities"] if clf.mode == "cnn" else {}
            if not p:
                continue
            preds[max((k for k in p if k in FER7_LABELS), key=lambda k: p[k])] += 1
        cap.release()

        if not preds:
            sin_cara += 1
            continue

        pred = preds.most_common(1)[0][0]
        real = c["emocion"]
        acierto = int(pred == real)
        confusion[real][pred] += 1

        franja = "20-29" if c["edad"] < 30 else ("30-49" if c["edad"] < 50 else "50+")
        for tabla_dst, clave in (
            (por_emocion, real), (por_intensidad, c["intensidad"]),
            (por_sexo, c["sexo"]), (por_etnia, c["etnia"]),
            (por_edad, franja), (por_actor, c["actor"]),
        ):
            tabla_dst[clave][0] += acierto
            tabla_dst[clave][1] += 1

        if i % 40 == 0:
            print(f"  {i}/{len(clips)}", flush=True)

    det.close()

    total = sum(v[1] for v in por_emocion.values())
    ok = sum(v[0] for v in por_emocion.values())
    print(f"\n{'='*52}")
    print(f"ACIERTO GLOBAL: {ok}/{total} ({ok/max(total,1)*100:.0f}%)")
    print(f"(azar entre 6 emociones de CREMA-D: ~17%)")
    if sin_cara:
        print(f"clips sin cara detectada: {sin_cara}")

    tabla("por emocion:", por_emocion)
    tabla("por intensidad:", por_intensidad)
    tabla("por sexo:", por_sexo)
    tabla("por etnia:", por_etnia, minimo=5)
    tabla("por edad:", por_edad)

    # La dispersión entre actores dice si esto generaliza o depende de la cara
    tasas = [v[0] / v[1] for v in por_actor.values() if v[1] >= 5]
    if tasas:
        print(f"\ndispersion entre los {len(tasas)} actores:")
        print(f"  peor actor  {min(tasas)*100:>3.0f}%")
        print(f"  mejor actor {max(tasas)*100:>3.0f}%")
        print(f"  media       {sum(tasas)/len(tasas)*100:>3.0f}%")

    print("\nmatriz de confusion (esperado -> detectado):")
    for real in sorted(confusion):
        fila = ", ".join(f"{d}:{n}" for d, n in confusion[real].most_common(4))
        print(f"  {real:<10} -> {fila}")


if __name__ == "__main__":
    main()
