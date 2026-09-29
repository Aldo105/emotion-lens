"""
¿Mejora la calibración neutra de la página el reconocimiento del CNN?

La página calibra ~30 s de cara en reposo al empezar la sesión, pero ese basal
solo llega a los blendshapes y a las etiquetas derivadas: el CNN recibe la
imagen cruda. Aquí se simula llevar esa calibración también al CNN.

Simulación: cada actor de CREMA-D graba las 12 frases en tono neutro. Las 11
que NO están en la evaluación (todo lo que no es IEO) suman ~27 s de cara
neutra, casi lo mismo que la calibración real. Con ellas se mide el perfil de
probabilidades que el modelo da a esa cara en reposo, y se corrigen los 220
clips del arnés (incluidos los neutros de IEO, que no se usaron para calibrar).

Métodos, fijados antes de ver ningún resultado:
  sin       - lo que hace hoy el sistema
  resta     - p' = max(0, p - basal), renormalizado (el que midió
              evaluate_baseline_cnn.py con el modelo viejo: +2 puntos)
  cociente  - p' ∝ p / basal, la corrección del sesgo propio de cada cara
              hacia una clase (basal con suelo 1e-3)

Se aplican a los tres modelos: producción (FER2013) y los candidatos B y C.
Voto por clip: la clase más frecuente entre los frames, como el arnés.

Aviso: son 9 miradas al mismo conjunto de evaluación. Una diferencia de 1-2
clips entre métodos es ruido; solo cuenta un salto claro y consistente.

Uso:
    python evaluate_calibracion.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))

from backend.app.services.emotion_classifier import EmotionClassifier, FER7_LABELS
from backend.app.services.face_detector import FaceDetector
from backend.ml.cremad_faces import VIDEO_BASE, _download
from backend.ml.eval_summary import cumple_criterios, resumir

CLIPS = AQUI / "cremad"
CALIB = AQUI / "cremad_calibracion"
SALIDA = RAIZ / "backend" / "ml" / "resultados" / "calibracion"
BASE_JSON = RAIZ / "backend" / "ml" / "resultados" / "baseline_fer2013.json"

FRASES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]
MODELOS = {
    "produccion": None,
    "B_ferplus_cremad": RAIZ / "backend/ml/saved_models/candidatos/ferplus_cremad.pt",
    "C_ferplus_cremad_sad15": RAIZ / "backend/ml/saved_models/candidatos/ferplus_cremad_sad15.pt",
}
METODOS = ("sin", "resta", "cociente")


def corregir(p: np.ndarray, basal: np.ndarray, metodo: str) -> np.ndarray | None:
    if metodo == "sin":
        return p
    if metodo == "resta":
        q = np.maximum(0.0, p - basal)
    else:
        q = p / np.maximum(basal, 1e-3)
    total = q.sum()
    return q / total if total > 1e-9 else None


def voto(frames: np.ndarray, basal: np.ndarray, metodo: str) -> str | None:
    votos = Counter()
    for p in frames:
        q = corregir(p, basal, metodo)
        if q is not None:
            votos[FER7_LABELS[int(np.argmax(q))]] += 1
    return votos.most_common(1)[0][0] if votos else None


def probs_por_modelo(ruta: Path, det, clfs: dict) -> dict[str, np.ndarray]:
    """Una matriz (frames x 7) por modelo; la cara se detecta una sola vez."""
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = {n: [] for n in clfs}
    j = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        j += 1
        if j % 3:                      # 1 de cada 3 frames, como el arnés
            continue
        d = det.detect(frame, timestamp_ms=int(j / fps * 1000))
        if d is None:
            continue
        face = EmotionClassifier.preprocess_face(frame, d["bbox"])
        if face is None:
            continue
        for n, clf in clfs.items():
            p = clf._predict_cnn(face)["probabilities"]
            out[n].append([p[k] for k in FER7_LABELS])
    cap.release()
    return {n: np.array(v) for n, v in out.items()}


def main():
    clips = json.loads((CLIPS / "indice.json").read_text(encoding="utf-8"))["clips"]
    actores = sorted({c["actor"] for c in clips})

    CALIB.mkdir(exist_ok=True)
    faltan = [(a, f) for a in actores for f in FRASES[1:]
              if not (CALIB / f"{a}_{f}_NEU_XX.flv").exists()]
    print(f"clips de calibración a descargar: {len(faltan)}")
    for a, f in faltan:
        nombre = f"{a}_{f}_NEU_XX.flv"
        try:
            _download(f"{VIDEO_BASE}/{nombre}", CALIB / nombre, min_bytes=10_000)
        except Exception as e:
            print(f"  [WARN] {nombre}: {e}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clfs = {}
    for n, ruta in MODELOS.items():
        clf = EmotionClassifier(model_path=str(ruta) if ruta else None, device=device)
        if clf.mode != "cnn":
            sys.exit(f"no se pudo cargar {n}")
        clfs[n] = clf
    det = FaceDetector(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5)

    # Perfil basal por actor y modelo: media de todos los frames neutros de calibración
    basal = {}
    for i, a in enumerate(actores, 1):
        acum = {n: [] for n in clfs}
        for f in FRASES[1:]:
            ruta = CALIB / f"{a}_{f}_NEU_XX.flv"
            if ruta.exists():
                for n, m in probs_por_modelo(ruta, det, clfs).items():
                    if len(m):
                        acum[n].append(m)
        for n in clfs:
            frames = np.concatenate(acum[n]) if acum[n] else np.empty((0, 7))
            basal[(a, n)] = frames.mean(axis=0) if len(frames) else None
        print(f"  basal {i}/{len(actores)} actor {a}: "
              f"{sum(len(x) for x in acum[next(iter(clfs))])} frames neutros", flush=True)

    registros = {(n, m): [] for n in clfs for m in METODOS}
    for i, c in enumerate(clips, 1):
        probs = probs_por_modelo(CLIPS / c["archivo"], det, clfs)
        for n in clfs:
            b = basal[(c["actor"], n)]
            if not len(probs[n]) or b is None:
                continue
            for m in METODOS:
                pred = voto(probs[n], b, m)
                if pred is not None:
                    registros[(n, m)].append({"actor": c["actor"], "real": c["emocion"],
                                              "pred": pred, "sexo": c["sexo"],
                                              "etnia": c["etnia"]})
        if i % 40 == 0:
            print(f"  {i}/{len(clips)}", flush=True)
    det.close()

    base = json.loads(BASE_JSON.read_text(encoding="utf-8"))
    SALIDA.mkdir(parents=True, exist_ok=True)
    emociones = ["angry", "disgust", "fear", "happy", "sad", "neutral"]
    print(f"\n{'modelo':<24}{'método':<10}{'global':>7}{'peor':>6}{'brecha':>7}  "
          + " ".join(f"{e[:5]:>6}" for e in emociones) + "  criterios")
    for (n, m), regs in registros.items():
        r = resumir(regs)
        r.update({"modelo": n, "metodo": m})
        (SALIDA / f"{n}_{m}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ok, fallos = cumple_criterios(base, r)
        tasas = " ".join(f"{r['por_emocion'].get(e, {}).get('tasa', 0)*100:>5.0f}%"
                         for e in emociones)
        print(f"{n:<24}{m:<10}{r['global']*100:>6.0f}%{r['actores']['peor']*100:>5.0f}%"
              f"{r['brecha_sexo']*100:>6.0f}p  {tasas}  "
              + ("PASA" if ok else "no: " + "; ".join(f.split(' ')[0] for f in fallos)))
    print(f"\nJSON por combinación en {SALIDA}")


if __name__ == "__main__":
    main()
