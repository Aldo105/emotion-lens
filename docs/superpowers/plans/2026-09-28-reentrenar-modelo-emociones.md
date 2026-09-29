# Reentrenar el CNN de emociones con datos públicos — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subir el accuracy *cross-actor* del CNN (hoy 36 % sobre CREMA-D, `sad` 12 %) reentrenándolo con etiquetas FER+ y caras de vídeo de CREMA-D, sin contaminar el arnés de evaluación.

**Architecture:** Cada reentrenamiento produce un *candidato* en `backend/ml/saved_models/candidatos/`, nunca sobre `emotion_cnn.pt`. Los datos nuevos se preparan con scripts aparte: `ferplus_labels.py` cambia las etiquetas de FER2013 por el voto de 10 anotadores, y `cremad_faces.py` extrae recortes de cara de 71 actores de CREMA-D que no están en la evaluación, con el mismo recorte que usa la inferencia. Un candidato solo sustituye al modelo en producción si pasa criterios fijados de antemano en `eval_summary.py`, medidos con `evaluate_cremad.py`.

**Tech Stack:** PyTorch + torchvision (MobileNetV2), HuggingFace `datasets`, OpenCV, MediaPipe (`FaceDetector` del propio repo), pytest. Entorno conda `emotion-lens`.

**Spec:** `notas-locales/plan-sensor-pulso.md` (sección "Pendiente para cuando llegue el disco/RAM: reentrenar el modelo") y `docs/investigacion-accuracy-sin-entrenar.md` (números de partida y §6: el sesgo demográfico solo se corrige con datos).

---

## Evaluación previa: qué datasets sirven y por qué

### Punto de partida

| Medición | Valor | Fuente |
|---|---|---|
| FER2013 test (in-domain) | 67 % (`sad` 54 %, `fear` 45 %) | `saved_models/training_report.json` |
| CREMA-D, 20 actores, 220 clips | **36 %**, peor actor 9 %, mejor actor 73 % | `references.py` |
| RAVDESS actor 1 | 46 %, `sad` 12 % | nota local |
| Mujeres frente a hombres | 40 % / 30 % | `references.py` |

**Diagnóstico:** el modelo generaliza mal entre caras (sesgo de identidad) y aprendió de etiquetas ruidosas: una sola persona etiquetó cada imagen de FER2013, a 48×48. Añadir más FER2013 no arregla ninguna de las dos cosas. Hacen falta **etiquetas mejores** y **más identidades, en el dominio real** (vídeo frontal hablando, como en una entrevista).

### Datasets candidatos

| Dataset | Tamaño | Acceso | Licencia | Veredicto |
|---|---|---|---|---|
| **FER+** (Microsoft) | Las mismas 35.9k imágenes de FER2013, cada una re-etiquetada por 10 anotadores | Libre: un CSV en GitHub | Etiquetas en repo MS (ver `LICENSE.md`); imágenes = FER2013 | **Fase 1.** Ataca el ruido de etiqueta sin descargar imágenes nuevas. En la literatura, pasar de etiquetas FER a FER+ sube el test de ~72 % a ~85 % en el mismo modelo. Coste casi nulo. |
| **CREMA-D** | 7,442 clips, **91 actores** (48 H / 43 M, 20–74 años, varias etnias) | Libre: GitHub | ODbL 1.0 + DbCL 1.0 | **Fase 1.** Es exactamente el dominio de uso (vídeo, cara frontal, hablando) y la fuente más diversa disponible sin firmar nada. Trae `FaceVote`: la emoción que perciben los evaluadores *solo con la cara*, que es mejor etiqueta que la emoción que se le pidió al actor. **Hay que excluir los 20 actores de evaluación.** |
| RAVDESS | 24 actores, vídeo | Libre: Zenodo | CC BY-NC-SA 4.0 | Fase 2, opcional. Poca diversidad demográfica y ~0.5 GB por actor. Hay que excluir al actor 1, que se usa en `evaluate.py`. |
| RAF-DB | ~15k imágenes básicas, en color, "in the wild" | EULA académica por email | No comercial | Fase 2, si se firma. Es muy buen complemento de FER+. |
| AffectNet | ~440k imágenes anotadas a mano | Acuerdo académico | No comercial | Fase 3. Es la palanca más grande, pero pesa decenas de GB, necesita el disco y la RAM nuevos y entrenar con ella en una GTX 1650 lleva días. **No usar los "mirrors" de HuggingFace o Kaggle: su licencia es dudosa.** |
| ExpW | ~91k imágenes | Descarga directa | Solo investigación | Fase 2. Etiquetas ruidosas, parecidas a FER2013. |
| CK+, JAFFE, KDEF | Cientos a pocos miles, posados | Registro | No comercial | Descartados: son pocos sujetos y muy homogéneos, lo que empeora justo el sesgo de identidad. |
| DFEW, FERV39k, MAFW, Aff-Wild2 | Vídeo "in the wild" | Acuerdo | No comercial | Fuera de alcance: exigen un modelo temporal. |

**Decisión:** este plan implementa **FER+ y CREMA-D**, que se pueden descargar hoy, sin firmar nada, y caben en esta máquina. Los datasets con acuerdo quedan para planes posteriores, cuando el usuario obtenga acceso. Casi todos son **no comerciales**, lo cual encaja con el uso educativo o de investigación que declara el README.

### Riesgos de este plan (se aceptan, conviene conocerlos)

1. **CREMA-D no tiene `surprise`.** Al añadirlo, esa clase queda relativamente menos representada. FER+ la sigue aportando, y el arnés CREMA-D tampoco la evalúa.
2. **El arnés se evalúa con la emoción pedida al actor, y el entrenamiento usa `FaceVote`.** Es deliberado: entrenar con la percepción visual reduce el ruido. Puede penalizar algo en clips donde el actor no consiguió expresar la emoción.
3. **Ganar en CREMA-D sin generalizar.** Con el reparto por actor, las caras de evaluación nunca se ven en entrenamiento, pero el estudio, la iluminación y el guion son los mismos. Por eso el Task 6 exige además una medida *end to end* con RAVDESS actor 1, que es otro estudio.
4. **Si el join de FER+ sale mal, el modelo entrena en silencio con etiquetas equivocadas.** El Task 3 tiene una puerta de verificación que detiene el plan en ese caso.
5. Este plan **no sustituye** la vía A1 de `investigacion-accuracy-sin-entrenar.md` (pesos publicados en AffectNet). El arnés del Task 2 (`--modelo` + JSON + criterios) sirve igual para comparar esos pesos, así que ambas vías se pueden decidir con el mismo número.

---

## Global Constraints

- Hardware actual: **16 GB de RAM (ampliada el 2026-09-28), GTX 1650 con 4 GB de VRAM, ~63 GB libres en C:**. Entrenar con `--num-workers 2` y sin el servidor de desarrollo abierto. El valor por defecto del script sigue siendo `NUM_WORKERS = 0`, que es el seguro.
- Python: `C:\Users\aldok\miniconda3\envs\emotion-lens\python.exe`, con `envs\emotion-lens`, `Library\bin` y `Scripts` antepuestos al PATH. Smart App Control bloquea hoy el `_ssl.pyd` de ese entorno, así que todo lo que use HTTPS (descargas, `datasets`) falla. Por eso `datasets` se importa dentro de las funciones que lo usan, no a nivel de módulo, para que los tests corran sin red.
- Los entrenamientos y la extracción de caras se lanzan **en primer plano**, no en background.
- `--batch-size 32` por defecto. Solo se sube a 64 si una época completa corre sin OOM en la GPU.
- **Nunca** se entrena con los actores de `data/test_videos/cremad/indice.json` (20 actores de CREMA-D) ni con RAVDESS `Actor_01`.
- **Nunca** se escribe en `backend/ml/saved_models/emotion_cnn.pt` salvo en el paso de promoción del Task 6.
- Los datos descargados van en `data/training/` y no se versionan.
- El preprocesado de inferencia no cambia de comportamiento: escala de grises replicada a 3 canales, 224×224, recorte 10 % hacia dentro, normalización ImageNet.
- Orden de etiquetas: `FER7_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]`.
- Commits **sin** `Co-Authored-By` ni trailers de Claude (preferencia del usuario).
- Todo resultado, bueno o malo, se registra en `backend/app/references.py`, como el resto de mediciones del proyecto.

## Review Focus

1. **Fuga de actores de evaluación al entrenamiento.** Un manifiesto que contenga un actor de `indice.json` debe hacer fallar el entrenamiento, no seguir adelante. Lo cubre el test `test_verify_manifest_rejects_eval_actor` (Task 4), y `ManifestDataset` llama a la verificación (Task 5).
2. **Join FER+ ↔ HF desalineado.** Si `sample_id` no es el índice original, la verificación debe fallar. Lo cubre el test `test_check_join_detects_misalignment` (Task 3).
3. **Recorte distinto entre entrenamiento e inferencia.** Tras el refactor, `preprocess_face` debe dar exactamente el mismo tensor que antes. Lo cubre el test `test_preprocess_face_matches_previous_implementation` (Task 1).
4. **Un entrenamiento que sobrescribe el modelo de producción.** Ningún `--name` puede resolver a `emotion_cnn.pt`. Lo cubre el test `test_candidate_path_never_production` (Task 5).
5. **Reanudar una extracción interrumpida y duplicar filas.** Los clips ya presentes en el manifiesto se saltan. Lo cubre el test `test_pending_clips_skips_done` (Task 4).

---

## File Structure

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `backend/app/services/emotion_classifier.py` | Modificar | Separar `crop_face_gray()` de `preprocess_face()` para que la extracción reutilice el recorte exacto de la inferencia |
| `backend/ml/eval_summary.py` | Crear | Resumen del arnés CREMA-D, comparación de candidato frente a línea base y criterios de promoción |
| `data/test_videos/evaluate_cremad.py` | Modificar | Aceptar `--modelo` y `--salida`, y volcar el JSON |
| `backend/ml/ferplus_labels.py` | Crear | Descargar FER+, calcular el voto mayoritario, hacer el join y verificarlo |
| `backend/ml/cremad_faces.py` | Crear | Reparto por actor, descarga, extracción de caras y manifiesto |
| `backend/ml/train_emotion_cnn.py` | Modificar | Argumentos de línea de comandos, etiquetas FER+, manifiesto extra, ruta de candidato y pesos por clase |
| `backend/ml/resultados/` | Crear | JSON de evaluación (versionado: es evidencia) |
| `.gitignore` | Modificar | Ignorar `data/training/` y `backend/ml/saved_models/candidatos/` |
| `backend/tests/test_face_crop.py`, `test_eval_summary.py`, `test_ferplus_labels.py`, `test_cremad_faces.py`, `test_train_helpers.py` | Crear | Tests |

---

### Task 1: Recorte de cara reutilizable

**Files:**
- Modify: `backend/app/services/emotion_classifier.py:188-227` (`preprocess_face`)
- Test: `backend/tests/test_face_crop.py`

**Interfaces:**
- Produces: `EmotionClassifier.crop_face_gray(frame_bgr: np.ndarray, bbox: dict) -> np.ndarray | None`, que devuelve un `uint8` de forma `(224, 224)`, o `None` si el recorte queda vacío. `preprocess_face` mantiene la firma y el resultado.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_face_crop.py
import cv2
import numpy as np

from backend.app.services.emotion_classifier import EmotionClassifier


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(360, 480, 3), dtype=np.uint8)


BBOX = {"x_min": 100, "y_min": 50, "x_max": 300, "y_max": 290}


def _old_preprocess(frame_bgr, bbox):
    """Verbatim copy of preprocess_face before the refactor."""
    x1 = max(0, bbox.get("x_min", 0))
    y1 = max(0, bbox.get("y_min", 0))
    x2 = bbox.get("x_max", 0)
    y2 = bbox.get("y_max", 0)
    inset_x = int((x2 - x1) * 0.10)
    inset_y = int((y2 - y1) * 0.10)
    x1, y1 = x1 + inset_x, y1 + inset_y
    x2, y2 = x2 - inset_x, y2 - inset_y
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    three = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    chw = three.transpose(2, 0, 1)
    chw = (chw - EmotionClassifier._CNN_MEAN) / EmotionClassifier._CNN_STD
    return chw[np.newaxis, ...].astype(np.float32)


def test_crop_face_gray_shape_and_dtype():
    gray = EmotionClassifier.crop_face_gray(_frame(), BBOX)
    assert gray.shape == (224, 224)
    assert gray.dtype == np.uint8


def test_crop_face_gray_empty_box_returns_none():
    assert EmotionClassifier.crop_face_gray(_frame(), {"x_min": 10, "y_min": 10, "x_max": 10, "y_max": 10}) is None


def test_preprocess_face_matches_previous_implementation():
    frame = _frame()
    np.testing.assert_array_equal(
        EmotionClassifier.preprocess_face(frame, BBOX), _old_preprocess(frame, BBOX))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_face_crop.py -v`
Expected: FAIL con `AttributeError: type object 'EmotionClassifier' has no attribute 'crop_face_gray'` (el test de equivalencia pasa ya).

- [ ] **Step 3: Write minimal implementation**

Sustituir el cuerpo de `preprocess_face` desde `x1 = max(0, ...)` hasta el `return`, y añadir antes el nuevo método. El docstring y los comentarios existentes (el del 10 % de recorte y el de la escala de grises) se trasladan tal cual a `crop_face_gray`:

```python
    @staticmethod
    def crop_face_gray(frame_bgr: np.ndarray, bbox: dict) -> np.ndarray | None:
        """
        Crop the detected face exactly as inference does and return it as a
        224x224 uint8 grayscale image. Shared with backend/ml/cremad_faces.py
        so training crops and live crops can never drift apart.
        """
        x1 = max(0, bbox.get("x_min", 0))
        y1 = max(0, bbox.get("y_min", 0))
        x2 = bbox.get("x_max", 0)
        y2 = bbox.get("y_max", 0)

        # (keep the existing "Tighten the detector's box by 10%" comment here)
        inset_x = int((x2 - x1) * 0.10)
        inset_y = int((y2 - y1) * 0.10)
        x1, y1 = x1 + inset_x, y1 + inset_y
        x2, y2 = x2 - inset_x, y2 - inset_y

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        size = EmotionClassifier._CNN_IMG_SIZE
        resized = cv2.resize(crop, (size, size))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def preprocess_face(frame_bgr: np.ndarray, bbox: dict) -> np.ndarray | None:
        """(keep the existing docstring)"""
        gray = EmotionClassifier.crop_face_gray(frame_bgr, bbox)
        if gray is None:
            return None
        three = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        chw = three.transpose(2, 0, 1)
        chw = (chw - EmotionClassifier._CNN_MEAN) / EmotionClassifier._CNN_STD
        return chw[np.newaxis, ...].astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `pytest backend/tests/test_face_crop.py -v` → 3 PASS. Luego `pytest backend/tests/` → todo PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/emotion_classifier.py backend/tests/test_face_crop.py
git commit -m "Split the face crop out of preprocess_face so training can reuse it"
```

---

### Task 2: Arnés CREMA-D comparable y criterios de promoción

**Files:**
- Create: `backend/ml/eval_summary.py`
- Modify: `data/test_videos/evaluate_cremad.py` (`main`, bucle de clips)
- Create: `backend/ml/resultados/baseline_fer2013.json` (salida del Step 6)
- Test: `backend/tests/test_eval_summary.py`

**Interfaces:**
- Produces:
  - `resumir(registros: list[dict]) -> dict`. Cada registro es `{"actor": str, "real": str, "pred": str, "sexo": str, "etnia": str}`. Devuelve `{"n", "global", "por_emocion", "por_sexo", "por_etnia", "actores": {"peor", "mejor", "media"}, "brecha_sexo", "confusion"}`. En `por_*`, cada valor es `{"aciertos", "total", "tasa"}`.
  - `cumple_criterios(base: dict, cand: dict) -> tuple[bool, list[str]]`
  - CLI: `python -m backend.ml.eval_summary BASE.json CAND.json`
  - `evaluate_cremad.py --modelo RUTA.pt --salida RUTA.json`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_eval_summary.py
from backend.ml.eval_summary import cumple_criterios, resumir


def _reg(actor, real, pred, sexo="Female", etnia="Caucasian"):
    return {"actor": actor, "real": real, "pred": pred, "sexo": sexo, "etnia": etnia}


def _registros():
    regs = []
    for i in range(5):  # actor A: 5/5, actor B: 1/5
        regs.append(_reg("A", "happy", "happy"))
        regs.append(_reg("B", "sad", "sad" if i == 0 else "neutral", sexo="Male"))
    return regs


def test_resumir_global_and_per_actor():
    r = resumir(_registros())
    assert r["n"] == 10
    assert r["global"] == 0.6
    assert r["actores"] == {"peor": 0.2, "mejor": 1.0, "media": 0.6}
    assert r["por_emocion"]["sad"] == {"aciertos": 1, "total": 5, "tasa": 0.2}
    assert r["brecha_sexo"] == 0.8
    assert r["confusion"]["sad"] == {"sad": 1, "neutral": 4}


def test_resumir_ignores_actors_with_fewer_than_5_clips_for_spread():
    regs = _registros() + [_reg("C", "angry", "happy")]
    assert resumir(regs)["actores"]["peor"] == 0.2


def _summary(glob, sad, happy, peor, brecha):
    return {"global": glob, "actores": {"peor": peor}, "brecha_sexo": brecha,
            "por_emocion": {"sad": {"tasa": sad}, "happy": {"tasa": happy}}}


def test_cumple_criterios_accepts_clear_improvement():
    base = _summary(0.36, 0.10, 0.95, 0.09, 0.10)
    cand = _summary(0.45, 0.30, 0.90, 0.15, 0.08)
    assert cumple_criterios(base, cand) == (True, [])


def test_cumple_criterios_lists_every_failure():
    base = _summary(0.36, 0.10, 0.95, 0.09, 0.10)
    cand = _summary(0.38, 0.20, 0.80, 0.05, 0.20)
    ok, fallos = cumple_criterios(base, cand)
    assert not ok
    assert len(fallos) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_eval_summary.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'backend.ml.eval_summary'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/ml/eval_summary.py
"""
Resumen del arnés CREMA-D y criterios de promoción de un candidato.

Los criterios se fijan aquí, antes de ver ningún resultado, para que elegir
el modelo no se convierta en buscar el número que convenga.

Uso:
    python -m backend.ml.eval_summary BASE.json CANDIDATO.json
"""
import json
import sys
from collections import Counter, defaultdict

MIN_CLIPS_POR_ACTOR = 5

# Criterios de promoción (fijados el 2026-09-28, antes de entrenar nada)
MEJORA_GLOBAL_MIN = 0.05      # +5 puntos sobre la línea base
MEJORA_SAD_MIN = 0.15         # el objetivo acordado: que 'sad' se detecte
HAPPY_MIN = 0.85              # no romper la única clase que ya funciona
BRECHA_SEXO_TOLERANCIA = 0.02 # la brecha mujeres/hombres no puede crecer


def _tasas(pares):
    return {k: {"aciertos": ok, "total": n, "tasa": round(ok / n, 4)}
            for k, (ok, n) in pares.items()}


def resumir(registros: list[dict]) -> dict:
    grupos = {c: defaultdict(lambda: [0, 0]) for c in ("real", "sexo", "etnia", "actor")}
    confusion = defaultdict(Counter)
    for r in registros:
        acierto = int(r["pred"] == r["real"])
        confusion[r["real"]][r["pred"]] += 1
        for campo, tabla in grupos.items():
            tabla[r[campo]][0] += acierto
            tabla[r[campo]][1] += 1

    n = len(registros)
    aciertos = sum(v[0] for v in grupos["real"].values())
    por_actor = [ok / t for ok, t in grupos["actor"].values() if t >= MIN_CLIPS_POR_ACTOR]
    por_sexo = _tasas(grupos["sexo"])
    tasas_sexo = [v["tasa"] for v in por_sexo.values()]
    return {
        "n": n,
        "global": round(aciertos / n, 4) if n else 0.0,
        "por_emocion": _tasas(grupos["real"]),
        "por_sexo": por_sexo,
        "por_etnia": _tasas(grupos["etnia"]),
        "actores": {
            "peor": round(min(por_actor), 4) if por_actor else 0.0,
            "mejor": round(max(por_actor), 4) if por_actor else 0.0,
            "media": round(sum(por_actor) / len(por_actor), 4) if por_actor else 0.0,
        },
        "brecha_sexo": round(max(tasas_sexo) - min(tasas_sexo), 4) if tasas_sexo else 0.0,
        "confusion": {real: dict(c) for real, c in confusion.items()},
    }


def _tasa(res, emocion):
    return res["por_emocion"].get(emocion, {}).get("tasa", 0.0)


def cumple_criterios(base: dict, cand: dict) -> tuple[bool, list[str]]:
    fallos = []
    if cand["global"] < base["global"] + MEJORA_GLOBAL_MIN:
        fallos.append(f"global {cand['global']:.2f} < {base['global']:.2f} + {MEJORA_GLOBAL_MIN}")
    if _tasa(cand, "sad") < _tasa(base, "sad") + MEJORA_SAD_MIN:
        fallos.append(f"sad {_tasa(cand, 'sad'):.2f} < {_tasa(base, 'sad'):.2f} + {MEJORA_SAD_MIN}")
    if _tasa(cand, "happy") < HAPPY_MIN:
        fallos.append(f"happy {_tasa(cand, 'happy'):.2f} < {HAPPY_MIN}")
    if cand["actores"]["peor"] < base["actores"]["peor"]:
        fallos.append(f"peor actor {cand['actores']['peor']:.2f} < {base['actores']['peor']:.2f}")
    if cand["brecha_sexo"] > base["brecha_sexo"] + BRECHA_SEXO_TOLERANCIA:
        fallos.append(f"brecha sexo {cand['brecha_sexo']:.2f} > {base['brecha_sexo']:.2f} + {BRECHA_SEXO_TOLERANCIA}")
    return (not fallos, fallos)


def main(argv):
    base, cand = (json.loads(open(p, encoding="utf-8").read()) for p in argv[:2])
    for clave in ("global", "brecha_sexo"):
        print(f"{clave:<12} {base[clave]:.2f} -> {cand[clave]:.2f}")
    print(f"{'peor actor':<12} {base['actores']['peor']:.2f} -> {cand['actores']['peor']:.2f}")
    for emo in sorted(base["por_emocion"]):
        print(f"{emo:<12} {_tasa(base, emo):.2f} -> {_tasa(cand, emo):.2f}")
    ok, fallos = cumple_criterios(base, cand)
    print("\nPROMOCIONABLE" if ok else "\nNO PROMOCIONABLE:\n  " + "\n  ".join(fallos))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_eval_summary.py -v` → 4 PASS

- [ ] **Step 5: Añadir `--modelo` y `--salida` a `evaluate_cremad.py`**

En los imports añadir `import argparse` y `from backend.ml.eval_summary import resumir`. Al principio de `main()`:

```python
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=None, help="checkpoint .pt (por defecto el de producción)")
    ap.add_argument("--salida", default=None, help="escribe el resumen JSON aquí")
    args = ap.parse_args()
```

Sustituir `clf = EmotionClassifier()` por `clf = EmotionClassifier(model_path=args.modelo)`. Antes del bucle, `registros = []`. Dentro del bucle, justo después de `confusion[real][pred] += 1`:

```python
        registros.append({"actor": c["actor"], "real": real, "pred": pred,
                          "sexo": c["sexo"], "etnia": c["etnia"]})
```

Al final de `main()`, después de imprimir la matriz de confusión:

```python
    if args.salida:
        res = resumir(registros)
        res["modelo"] = args.modelo or "produccion"
        res["sin_cara"] = sin_cara
        Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.salida).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nresumen -> {args.salida}")
```

- [ ] **Step 6: Medir la línea base**

Run (desde la raíz del repo, en primer plano y sin el servidor abierto):
`python data/test_videos/evaluate_cremad.py --salida backend/ml/resultados/baseline_fer2013.json`
Expected: `ACIERTO GLOBAL` ≈ 36 % (80/220 ± pocos clips), igual que en `references.py`. Si difiere en más de 3 puntos, **parar**: el arnés o el modelo han cambiado y la comparación no vale.

- [ ] **Step 7: Commit**

```bash
git add backend/ml/eval_summary.py backend/tests/test_eval_summary.py data/test_videos/evaluate_cremad.py backend/ml/resultados/baseline_fer2013.json
git commit -m "Dump the CREMA-D harness to JSON and fix the promotion criteria up front"
```

---

### Task 3: Etiquetas FER+ con verificación del join

**Files:**
- Create: `backend/ml/ferplus_labels.py`
- Modify: `.gitignore` (añadir `data/training/` y `backend/ml/saved_models/candidatos/`)
- Test: `backend/tests/test_ferplus_labels.py`

**Interfaces:**
- Consumes: `FER7_LABELS` de `backend.app.services.emotion_classifier`.
- Produces:
  - `majority_label(votes: dict[str, int]) -> int | None`
  - `load_votes(csv_path) -> list[dict]`, con filas `{"usage": str, "votes": dict[str, int]}` en el orden original.
  - `index_from_sample_id(sample_id: str) -> int`
  - `labels_for_split(hf_split, rows) -> list[int | None]`
  - `check_join(hf_splits: dict, rows) -> dict`, que devuelve `{"n", "usage_match", "label_agreement", "dropped"}`
  - `join_ok(stats: dict) -> bool`
  - `ensure_csv() -> Path`
  - CLI: `python -m backend.ml.ferplus_labels --verificar`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ferplus_labels.py
import pytest

from backend.ml.ferplus_labels import (
    check_join, index_from_sample_id, join_ok, labels_for_split, load_votes, majority_label,
)

COLS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "nf"]


def _votes(**kw):
    return {c: kw.get(c, 0) for c in COLS}


def test_majority_clear_winner_maps_to_fer7_index():
    assert majority_label(_votes(sadness=7, neutral=3)) == 4  # sad


def test_single_stray_votes_are_dropped_before_majority():
    # 5 sadness vs 4 neutral + 1 anger: stray anger vote removed -> 5/9 > half
    assert majority_label(_votes(sadness=5, neutral=4, anger=1)) == 4


def test_no_majority_returns_none():
    assert majority_label(_votes(sadness=4, neutral=4, anger=2)) is None


def test_contempt_unknown_and_not_a_face_are_dropped():
    assert majority_label(_votes(contempt=8, neutral=2)) is None
    assert majority_label(_votes(unknown=6, neutral=4)) is None
    assert majority_label(_votes(nf=10)) is None
    assert majority_label(_votes()) is None


def test_index_from_sample_id():
    assert index_from_sample_id("fer2013_006289") == 6289
    with pytest.raises(ValueError):
        index_from_sample_id("affectnet_12")


def test_load_votes_reads_header_names_not_positions(tmp_path):
    p = tmp_path / "fer2013new.csv"
    p.write_text("Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF\n"
                 "Training,fer0000000.png,4,0,0,1,5,0,0,0,0,0\n"
                 "PublicTest,,0,0,0,0,0,0,0,0,0,10\n", encoding="utf-8")
    rows = load_votes(p)
    assert rows[0]["usage"] == "Training"
    assert rows[0]["votes"]["anger"] == 5
    assert rows[1]["votes"]["nf"] == 10


def _rows():
    return [
        {"usage": "Training", "votes": _votes(happiness=10)},
        {"usage": "Training", "votes": _votes(sadness=9, neutral=1)},
        {"usage": "PublicTest", "votes": _votes(nf=10)},
    ]


def test_labels_for_split_follows_sample_id():
    split = {"sample_id": ["fer2013_000001", "fer2013_000000"]}
    assert labels_for_split(split, _rows()) == [4, 3]


def test_check_join_passes_when_aligned():
    split = {"sample_id": ["fer2013_000000", "fer2013_000001", "fer2013_000002"],
             "usage": ["Training", "Training", "PublicTest"],
             "emotion": [3, 4, 6]}
    stats = check_join({"train": split}, _rows())
    assert stats["usage_match"] == 1.0
    assert stats["label_agreement"] == 1.0
    assert stats["dropped"] == pytest.approx(1 / 3)


def test_check_join_detects_misalignment():
    split = {"sample_id": ["fer2013_000002", "fer2013_000000", "fer2013_000001"],
             "usage": ["Training", "Training", "PublicTest"],
             "emotion": [3, 4, 6]}
    assert not join_ok(check_join({"train": split}, _rows()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_ferplus_labels.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'backend.ml.ferplus_labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/ml/ferplus_labels.py
"""
FER+ labels for the FER2013 images already used in training.

FER2013 was labelled by one person per image; FER+ (Microsoft) re-labelled
the same images with 10 crowd taggers each. fer2013new.csv has one row per
row of the original fer2013.csv, in the same order. The HF mirror we train
from (abhilash88/fer2013-enhanced) re-splits the data, but its sample_id
("fer2013_006289") appears to encode that original row index. "Appears" is
not good enough to train on, so --verificar checks it before anything uses
these labels.

Usage:
    python -m backend.ml.ferplus_labels --verificar
"""
import argparse
import csv
import sys
import urllib.request
from pathlib import Path

from backend.app.services.emotion_classifier import FER7_LABELS

FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "training" / "ferplus" / "fer2013new.csv"

VOTE_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger",
                "disgust", "fear", "contempt", "unknown", "nf"]
FERPLUS_TO_FER7 = {"anger": "angry", "disgust": "disgust", "fear": "fear",
                   "happiness": "happy", "sadness": "sad",
                   "surprise": "surprise", "neutral": "neutral"}


def majority_label(votes: dict[str, int]) -> int | None:
    """Microsoft's 'majority' mode: discard single stray votes, then keep the
    image only if one class holds more than half of what remains. contempt,
    unknown and not-a-face have no FER7 equivalent and are dropped."""
    cleaned = {k: (v if v > 1 else 0) for k, v in votes.items()}
    total = sum(cleaned.values())
    if total == 0:
        return None
    top = max(cleaned, key=cleaned.get)
    if cleaned[top] * 2 <= total:
        return None
    name = FERPLUS_TO_FER7.get(top)
    return FER7_LABELS.index(name) if name else None


def load_votes(csv_path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            norm = {k.strip().lower(): (v or "").strip() for k, v in raw.items()}
            rows.append({"usage": norm.get("usage", ""),
                         "votes": {c: int(norm.get(c) or 0) for c in VOTE_COLUMNS}})
    return rows


def index_from_sample_id(sample_id: str) -> int:
    prefix, _, num = sample_id.rpartition("_")
    if prefix != "fer2013" or not num.isdigit():
        raise ValueError(f"unexpected sample_id: {sample_id!r}")
    return int(num)


def labels_for_split(hf_split, rows) -> list[int | None]:
    return [majority_label(rows[index_from_sample_id(s)]["votes"])
            for s in hf_split["sample_id"]]


def check_join(hf_splits: dict, rows) -> dict:
    n = usage_ok = kept = agree = 0
    for split in hf_splits.values():
        for sid, usage, emotion in zip(split["sample_id"], split["usage"], split["emotion"]):
            row = rows[index_from_sample_id(sid)]
            n += 1
            usage_ok += int(row["usage"] == usage)
            label = majority_label(row["votes"])
            if label is not None:
                kept += 1
                agree += int(label == emotion)
    return {"n": n,
            "usage_match": usage_ok / n if n else 0.0,
            "label_agreement": agree / kept if kept else 0.0,
            "dropped": (n - kept) / n if n else 0.0}


def join_ok(stats: dict) -> bool:
    # A correct join puts every image in its original Usage split, and FER+
    # re-labels only a minority of images; a shuffled join agrees at chance.
    return stats["usage_match"] >= 0.999 and 0.5 <= stats["label_agreement"] <= 0.97


def ensure_csv() -> Path:
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading {FERPLUS_URL}")
        urllib.request.urlretrieve(FERPLUS_URL, CSV_PATH)
    return CSV_PATH


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true")
    args = ap.parse_args(argv)
    if not args.verificar:
        ap.print_help()
        return 0
    from datasets import load_dataset
    raw = load_dataset("abhilash88/fer2013-enhanced")
    rows = load_votes(ensure_csv())
    print(f"[INFO] fer2013new.csv rows={len(rows)}")
    stats = check_join(dict(raw), rows)
    print(f"[INFO] {stats}")
    if not join_ok(stats):
        print("[ERROR] FER+ join does not line up -- do NOT train with --labels ferplus")
        return 1
    print("[OK] FER+ join verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Añadir al final de `.gitignore`:

```gitignore
# Datos de entrenamiento descargados (FER+, caras de CREMA-D) y candidatos
data/training/
backend/ml/saved_models/candidatos/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_ferplus_labels.py -v` → 9 PASS

- [ ] **Step 5: Puerta de verificación con los datos reales**

Run: `python -m backend.ml.ferplus_labels --verificar`
Expected: `usage_match` ≥ 0.999, `label_agreement` entre 0.5 y 0.97, `dropped` en torno a 0.1, y `[OK] FER+ join verified`.
**Si sale `[ERROR]`: parar el plan aquí** y avisar al usuario. La alternativa es bajar `fer2013.csv` de Kaggle (necesita cuenta) y hacer el join por posición de fila, pero eso ya es otro plan.

- [ ] **Step 6: Commit**

```bash
git add backend/ml/ferplus_labels.py backend/tests/test_ferplus_labels.py .gitignore
git commit -m "Add FER+ majority labels with a join check against the HF mirror"
```

---

### Task 4: Caras de CREMA-D para entrenamiento, sin actores de evaluación

**Files:**
- Create: `backend/ml/cremad_faces.py`
- Test: `backend/tests/test_cremad_faces.py`

**Interfaces:**
- Consumes: `EmotionClassifier.crop_face_gray` (Task 1), `FaceDetector` y `FER7_LABELS`.
- Produces:
  - `EVAL_INDEX: Path`, que apunta a `data/test_videos/cremad/indice.json`.
  - `face_vote_label(vote: str) -> int | None`
  - `eval_actors(index_path) -> set[str]`, que lanza `FileNotFoundError` si falta el índice.
  - `split_actors(actors, excluded, n_val=10, seed=0) -> tuple[list[str], list[str]]`
  - `pick_frames(n_frames: int, k: int = 5, start: float = 0.4, end: float = 0.8) -> list[int]`
  - `read_summary(path) -> list[dict]`, con filas `{"clip", "actor", "label"}`.
  - `pending_clips(clips: list[dict], manifest_rows: list[dict]) -> list[dict]`
  - `verify_manifest(rows, excluded: set[str]) -> None`, que lanza `ValueError` si aparece un actor excluido.
  - CLI: `python -m backend.ml.cremad_faces [--max-actors N] [--frames 5] [--delete-videos]`. Escribe `data/training/cremad/manifest.csv` con las columnas `ruta,label,actor,split,clip`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_cremad_faces.py
import json

import pytest

from backend.ml.cremad_faces import (
    eval_actors, face_vote_label, pending_clips, pick_frames, read_summary,
    split_actors, verify_manifest,
)


def test_face_vote_label_maps_single_votes_and_rejects_ties():
    assert face_vote_label("S") == 4
    assert face_vote_label("N") == 6
    assert face_vote_label("A:S") is None
    assert face_vote_label("") is None


def test_eval_actors_reads_index(tmp_path):
    p = tmp_path / "indice.json"
    p.write_text(json.dumps({"clips": [{"actor": "1073"}, {"actor": "1002"}, {"actor": "1073"}]}))
    assert eval_actors(p) == {"1073", "1002"}


def test_eval_actors_missing_index_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_cremad"):
        eval_actors(tmp_path / "nope.json")


def test_split_actors_disjoint_deterministic_and_excludes():
    actors = [str(1001 + i) for i in range(91)]
    excluded = {"1002", "1019", "1020"}
    train, val = split_actors(actors, excluded, n_val=10, seed=0)
    assert len(val) == 10 and len(train) == 91 - 3 - 10
    assert not (set(train) & set(val))
    assert not ((set(train) | set(val)) & excluded)
    assert (train, val) == split_actors(actors, excluded, n_val=10, seed=0)


def test_pick_frames_middle_of_clip():
    assert pick_frames(100, k=5) == [40, 48, 56, 64, 72]
    assert pick_frames(5, k=5) == [2, 3]
    assert pick_frames(0) == []


def test_read_summary_parses_actor_and_face_vote(tmp_path):
    p = tmp_path / "summaryTable.csv"
    p.write_text('"","FileName","VoiceVote","VoiceLevel","FaceVote","FaceLevel","MultiModalVote","MultiModalLevel"\n'
                 '"1","1001_IEO_NEU_XX","N","69.1","N","92.22","N","64.78"\n'
                 '"2","1001_IEO_HAP_LO","N","71.67","H:N","57","H","57.38"\n', encoding="utf-8")
    assert read_summary(p) == [
        {"clip": "1001_IEO_NEU_XX", "actor": "1001", "label": 6},
        {"clip": "1001_IEO_HAP_LO", "actor": "1001", "label": None},
    ]


def test_pending_clips_skips_done():
    clips = [{"clip": "a"}, {"clip": "b"}, {"clip": "c"}]
    done = [{"clip": "b"}, {"clip": "b"}]
    assert pending_clips(clips, done) == [{"clip": "a"}, {"clip": "c"}]


def test_verify_manifest_rejects_eval_actor():
    verify_manifest([{"actor": "1001"}], {"1073"})
    with pytest.raises(ValueError, match="1073"):
        verify_manifest([{"actor": "1001"}, {"actor": "1073"}], {"1073"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_cremad_faces.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'backend.ml.cremad_faces'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/ml/cremad_faces.py
"""
Face crops from CREMA-D for training, from actors the evaluation never sees.

data/test_videos/evaluate_cremad.py scores the model on 20 CREMA-D actors
(listed in data/test_videos/cremad/indice.json). Training on any of them
would make that number meaningless, so they are excluded here and again
when the manifest is loaded (train_emotion_cnn.ManifestDataset).

Labels are CREMA-D's FaceVote -- what raters perceived from the face alone --
not the emotion the actor was asked to portray. Clips where raters split
("A:S") are skipped. Frames come from 40-80% of each clip: the onset is
still neutral and the expression peaks mid-sentence.

Licence: ODbL 1.0 / DbCL 1.0. Downloads stay in data/training/ (git-ignored).

Usage:
    python -m backend.ml.cremad_faces --max-actors 2     # smoke test
    python -m backend.ml.cremad_faces --delete-videos    # full run
"""
import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

import cv2

from backend.app.services.emotion_classifier import FER7_LABELS, EmotionClassifier

ROOT = Path(__file__).resolve().parents[2]
EVAL_INDEX = ROOT / "data" / "test_videos" / "cremad" / "indice.json"
OUT = ROOT / "data" / "training" / "cremad"
MANIFEST = OUT / "manifest.csv"
SUMMARY_URL = ("https://raw.githubusercontent.com/CheyneyComputerScience/"
               "CREMA-D/master/processedResults/summaryTable.csv")
VIDEO_BASE = ("https://media.githubusercontent.com/media/"
              "CheyneyComputerScience/CREMA-D/master/VideoFlash")
FACE_VOTE_TO_FER7 = {"A": "angry", "D": "disgust", "F": "fear",
                     "H": "happy", "N": "neutral", "S": "sad"}
FIELDS = ["ruta", "label", "actor", "split", "clip"]


def face_vote_label(vote: str) -> int | None:
    name = FACE_VOTE_TO_FER7.get(vote.strip())
    return FER7_LABELS.index(name) if name else None


def eval_actors(index_path) -> set[str]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} missing -- run data/test_videos/download_cremad.py --actores 20 "
            "first, otherwise evaluation actors cannot be excluded")
    clips = json.loads(index_path.read_text(encoding="utf-8"))["clips"]
    return {c["actor"] for c in clips}


def split_actors(actors, excluded, n_val=10, seed=0):
    pool = sorted(set(actors) - set(excluded))
    random.Random(seed).shuffle(pool)
    return sorted(pool[n_val:]), sorted(pool[:n_val])


def pick_frames(n_frames: int, k: int = 5, start: float = 0.4, end: float = 0.8) -> list[int]:
    if n_frames <= 0:
        return []
    a = int(n_frames * start)
    b = max(a + 1, int(n_frames * end))
    if b - a <= k:
        return list(range(a, b))
    step = (b - a) / k
    return [a + int(i * step) for i in range(k)]


def read_summary(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{"clip": r["FileName"], "actor": r["FileName"].split("_")[0],
                 "label": face_vote_label(r["FaceVote"])}
                for r in csv.DictReader(f)]


def pending_clips(clips: list[dict], manifest_rows: list[dict]) -> list[dict]:
    done = {r["clip"] for r in manifest_rows}
    return [c for c in clips if c["clip"] not in done]


def verify_manifest(rows, excluded: set[str]) -> None:
    leaked = sorted({r["actor"] for r in rows} & set(excluded))
    if leaked:
        raise ValueError(f"manifest contains evaluation actors: {leaked}")


def _read_frames(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def build(max_actors=None, frames_per_clip=5, delete_videos=False):
    from backend.app.services.face_detector import FaceDetector

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "caras").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    summary_path = OUT / "summaryTable.csv"
    if not summary_path.exists():
        urllib.request.urlretrieve(SUMMARY_URL, summary_path)

    clips = [c for c in read_summary(summary_path) if c["label"] is not None]
    excluded = eval_actors(EVAL_INDEX)
    train, val = split_actors({c["actor"] for c in clips}, excluded)
    if max_actors:
        train, val = train[:max_actors], val[:1]
    split_of = {a: "train" for a in train} | {a: "val" for a in val}
    clips = [c for c in clips if c["actor"] in split_of]

    existing = []
    if MANIFEST.exists():
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    verify_manifest(existing, excluded)
    todo = pending_clips(clips, existing)
    print(f"[INFO] actors train={len(train)} val={len(val)} excluded={len(excluded)} "
          f"clips pending={len(todo)} done={len({r['clip'] for r in existing})}")

    det = FaceDetector(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5)
    new_file = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, c in enumerate(todo, 1):
            video = OUT / "videos" / f"{c['clip']}.flv"
            if not video.exists():
                try:
                    urllib.request.urlretrieve(f"{VIDEO_BASE}/{c['clip']}.flv", video)
                except Exception as e:  # a missing clip must not kill a 6000-clip run
                    print(f"[WARN] {c['clip']}: {e}")
                    continue
            frames = _read_frames(video)
            for idx in pick_frames(len(frames), k=frames_per_clip):
                d = det.detect(frames[idx])
                gray = EmotionClassifier.crop_face_gray(frames[idx], d["bbox"]) if d else None
                if gray is None:
                    continue
                rel = f"caras/{c['clip']}_{idx:03d}.png"
                cv2.imwrite(str(OUT / rel), gray)
                writer.writerow({"ruta": rel, "label": c["label"], "actor": c["actor"],
                                 "split": split_of[c["actor"]], "clip": c["clip"]})
            f.flush()
            if delete_videos:
                video.unlink(missing_ok=True)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    det.close()
    print(f"[OK] manifest -> {MANIFEST}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-actors", type=int, default=None)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--delete-videos", action="store_true")
    args = ap.parse_args(argv)
    build(args.max_actors, args.frames, args.delete_videos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Nota: un clip sin ninguna cara detectada no escribe filas, así que se reintentará en la siguiente ejecución. Es barato: un clip que falla siempre cuesta unos segundos por pasada.

- [ ] **Step 4: Run tests**

Run: `pytest backend/tests/test_cremad_faces.py -v` → 8 PASS

- [ ] **Step 5: Humo con datos reales**

Run: `python -m backend.ml.cremad_faces --max-actors 2`
Expected: `actors train=2 val=1 excluded=20`, y aparecen pngs en `data/training/cremad/caras/`. Abrir 3 o 4 a mano y comprobar que son caras en escala de grises recortadas como en la inferencia. Luego ejecutar:

```bash
python -c "import csv;from backend.ml.cremad_faces import *;verify_manifest(list(csv.DictReader(open(MANIFEST))), eval_actors(EVAL_INDEX));print('sin fuga')"
```

Expected: `sin fuga`. Borrar `data/training/cremad/` antes de la extracción completa es opcional: `--max-actors` usa un prefijo del mismo reparto, así que las filas del humo son válidas.

- [ ] **Step 6: Commit**

```bash
git add backend/ml/cremad_faces.py backend/tests/test_cremad_faces.py
git commit -m "Extract CREMA-D training faces from actors outside the evaluation set"
```

---

### Task 5: Script de entrenamiento con candidatos, FER+ y manifiesto extra

**Files:**
- Modify: `backend/ml/train_emotion_cnn.py` (config, `FER2013Dataset`, `compute_class_weights`, `main`)
- Test: `backend/tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `load_votes`, `labels_for_split` y `ensure_csv` (Task 3); el manifiesto CSV del Task 4, con columnas `ruta,label,actor,split,clip` y `ruta` relativa al directorio del manifiesto. Del Task 4, `verify_manifest(rows, eval_actors: set[str])` y `eval_actors(path)`.
- Produces:
  - `parse_args(argv=None) -> argparse.Namespace`, con los campos `name`, `labels` (`"fer"` o `"ferplus"`), `extra_manifest`, `init_from`, `sad_weight_mult`, `num_workers`, `batch_size` y `max_epochs`.
  - `candidate_paths(name: str) -> tuple[str, str]`, que da `(.pt, .json)`.
  - `class_weights_from_labels(labels: list[int], sad_mult: float = 1.0) -> torch.Tensor`
  - `read_manifest(path, split: str) -> list[dict]`
  - `ManifestDataset(manifest_path, split, train)`
  - `FER2013Dataset(hf_split, train, labels=None)`, que ahora filtra las etiquetas `None`.
  - `select_val_acc(fer_val_acc: float, extra_val_acc: float | None) -> float`
  - Mantiene los nombres que importa `evaluate_model.py`: `FER2013Dataset`, `FER7_LABELS`, `OUTPUT_MODEL_PATH`, `BATCH_SIZE`, `build_model` y `run_epoch`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_train_helpers.py
import cv2
import numpy as np
import pytest

from backend.ml import train_emotion_cnn as T


class FakeSplit:
    def __init__(self, emotions):
        self.rows = [{"image": np.full((48, 48), 128, np.uint8), "emotion": e} for e in emotions]

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        return self.rows[key]

    def __len__(self):
        return len(self.rows)


def test_candidate_path_never_production():
    for name in ("emotion_cnn", "../emotion_cnn", "x"):
        pt, js = T.candidate_paths(name)
        assert pt != T.OUTPUT_MODEL_PATH
        assert "candidatos" in pt and pt.endswith(".pt") and js.endswith(".json")


def test_class_weights_inverse_frequency_and_sad_multiplier():
    labels = [0] * 10 + [4] * 5 + [3] * 5 + [1, 2, 5, 6]
    w = T.class_weights_from_labels(labels)
    assert w[0] < w[4]
    w_sad = T.class_weights_from_labels(labels, sad_mult=2.0)
    assert float(w_sad[4]) == pytest.approx(2 * float(w[4]))
    assert float(w_sad[0]) == pytest.approx(float(w[0]))


def test_fer_dataset_drops_none_labels():
    ds = T.FER2013Dataset(FakeSplit([0, 1, 2]), train=False, labels=[5, None, 3])
    assert len(ds) == 2
    assert ds.labels == [5, 3]
    x, y = ds[1]
    assert tuple(x.shape) == (3, 224, 224) and y == 3


def _manifest(tmp_path):
    (tmp_path / "caras").mkdir()
    cv2.imwrite(str(tmp_path / "caras" / "a.png"), np.full((224, 224), 90, np.uint8))
    cv2.imwrite(str(tmp_path / "caras" / "b.png"), np.full((224, 224), 90, np.uint8))
    p = tmp_path / "manifest.csv"
    p.write_text("ruta,label,actor,split,clip\n"
                 "caras/a.png,4,1001,train,1001_DFA_SAD_XX\n"
                 "caras/b.png,3,1002,val,1002_DFA_HAP_XX\n", encoding="utf-8")
    return p


def test_read_manifest_filters_split(tmp_path):
    rows = T.read_manifest(_manifest(tmp_path), "train")
    assert [r["actor"] for r in rows] == ["1001"]


def test_manifest_dataset_loads_gray_png_as_3ch(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "load_eval_actors", lambda: set())
    ds = T.ManifestDataset(_manifest(tmp_path), "train", train=False)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 224, 224) and y == 4


def test_manifest_dataset_refuses_eval_actor(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "load_eval_actors", lambda: {"1001"})
    with pytest.raises(ValueError, match="1001"):
        T.ManifestDataset(_manifest(tmp_path), "train", train=False)


def test_manifest_dataset_empty_split_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "load_eval_actors", lambda: set())
    with pytest.raises(ValueError, match="no rows"):
        T.ManifestDataset(_manifest(tmp_path), "test", train=False)


def test_select_val_acc_prefers_target_domain():
    assert T.select_val_acc(0.7, None) == 0.7
    assert T.select_val_acc(0.7, 0.4) == 0.4


def test_parse_args_defaults():
    a = T.parse_args(["--name", "x"])
    assert (a.labels, a.extra_manifest, a.sad_weight_mult, a.num_workers) == ("fer", None, 1.0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_train_helpers.py -v`
Expected: FAIL con `AttributeError: module 'backend.ml.train_emotion_cnn' has no attribute 'candidate_paths'`

- [ ] **Step 3: Write minimal implementation**

**3a.** Imports: añadir `import argparse`, `import csv`, `import cv2` y `from torch.utils.data import ConcatDataset`. Debajo de `OUTPUT_REPORT_PATH`:

```python
CANDIDATES_DIR = os.path.join(SCRIPT_DIR, "saved_models", "candidatos")


def candidate_paths(name: str) -> tuple[str, str]:
    """Every run writes here -- promoting a candidate to OUTPUT_MODEL_PATH
    is a deliberate manual step after evaluate_cremad.py (see the plan)."""
    safe = os.path.basename(name.replace("\\", "/")) or "run"
    return (os.path.join(CANDIDATES_DIR, f"{safe}.pt"),
            os.path.join(CANDIDATES_DIR, f"{safe}.json"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train an emotion CNN candidate")
    p.add_argument("--name", required=True, help="candidate name -> saved_models/candidatos/<name>.pt")
    p.add_argument("--labels", choices=["fer", "ferplus"], default="fer")
    p.add_argument("--extra-manifest", default=None, help="manifest.csv from backend.ml.cremad_faces")
    p.add_argument("--init-from", default=None, help="checkpoint to start from instead of ImageNet")
    p.add_argument("--sad-weight-mult", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return p.parse_args(argv)
```

**3b.** Sustituir la clase `FER2013Dataset` por:

```python
def make_transform(train: bool):
    steps = [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3),
             transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class FER2013Dataset(Dataset):
    """Wraps the HF FER2013 split; converts 48x48 grayscale -> 3ch tensor.
    `labels` overrides the split's own (e.g. FER+); None entries are dropped."""

    def __init__(self, hf_split, train: bool, labels=None):
        self.ds = hf_split
        if labels is None:
            labels = list(hf_split["emotion"])
        self.indices = [i for i, lbl in enumerate(labels) if lbl is not None]
        self.labels = [int(labels[i]) for i in self.indices]
        self.transform = make_transform(train)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.ds[self.indices[idx]]
        img = np.array(row["image"], dtype=np.uint8)  # (48, 48)
        return self.transform(img), self.labels[idx]


def load_eval_actors() -> set[str]:
    from backend.ml.cremad_faces import EVAL_INDEX, eval_actors
    return eval_actors(EVAL_INDEX)


def read_manifest(path, split: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class ManifestDataset(Dataset):
    """Face crops written by backend/ml/cremad_faces.py: 224x224 grayscale
    PNGs, cut with EmotionClassifier.crop_face_gray exactly like inference."""

    def __init__(self, manifest_path, split: str, train: bool):
        from backend.ml.cremad_faces import verify_manifest
        rows = read_manifest(manifest_path, split)
        if not rows:
            raise ValueError(f"manifest {manifest_path} has no rows for split {split!r}")
        verify_manifest(rows, load_eval_actors())
        base = os.path.dirname(os.path.abspath(manifest_path))
        self.paths = [os.path.join(base, r["ruta"]) for r in rows]
        self.labels = [int(r["label"]) for r in rows]
        self.transform = make_transform(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        return self.transform(img), self.labels[idx]
```

**3c.** Sustituir `compute_class_weights` por:

```python
def class_weights_from_labels(labels, sad_mult: float = 1.0) -> torch.Tensor:
    """Inverse-frequency class weights -- 'disgust' is ~1.5% of FER2013.
    sad_mult deliberately over-weights 'sad', the class we agreed must work."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(FER7_LABELS))
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    weights[FER7_LABELS.index("sad")] *= sad_mult
    print(f"[INFO] Class counts: {dict(zip(FER7_LABELS, counts.tolist()))}")
    print(f"[INFO] Class weights: {dict(zip(FER7_LABELS, np.round(weights, 3).tolist()))}")
    return torch.tensor(weights, dtype=torch.float32)


def select_val_acc(fer_val_acc: float, extra_val_acc: float | None) -> float:
    """Early stopping follows the held-out CREMA-D actors when present: that
    is the domain the app runs in, FER2013 val is only a proxy."""
    return fer_val_acc if extra_val_acc is None else extra_val_acc
```

**3d.** Sustituir `main()` completo por:

```python
def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    model_path, report_path = candidate_paths(args.name)
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    print("[INFO] Loading FER2013 (abhilash88/fer2013-enhanced) from HuggingFace Hub...")
    raw = load_dataset("abhilash88/fer2013-enhanced")
    labels = {s: None for s in ("train", "validation", "test")}
    if args.labels == "ferplus":
        from backend.ml.ferplus_labels import ensure_csv, labels_for_split, load_votes
        rows = load_votes(ensure_csv())
        labels = {s: labels_for_split(raw[s], rows) for s in labels}

    train_parts = [FER2013Dataset(raw["train"], True, labels["train"])]
    fer_val = FER2013Dataset(raw["validation"], False, labels["validation"])
    fer_test = FER2013Dataset(raw["test"], False, labels["test"])
    extra_val = None
    if args.extra_manifest:
        train_parts.append(ManifestDataset(args.extra_manifest, "train", True))
        extra_val = ManifestDataset(args.extra_manifest, "val", False)
    train_ds = ConcatDataset(train_parts)
    print(f"[INFO] train={len(train_ds)} fer_val={len(fer_val)} "
          f"extra_val={len(extra_val) if extra_val else 0} fer_test={len(fer_test)}")

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"))

    train_loader = loader(train_ds, True)
    fer_val_loader, fer_test_loader = loader(fer_val, False), loader(fer_test, False)
    extra_val_loader = loader(extra_val, False) if extra_val else None

    model = build_model(device)
    all_labels = [lbl for part in train_parts for lbl in part.labels]
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(all_labels, args.sad_weight_mult).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def evaluate():
        _, fer_acc = run_epoch(model, fer_val_loader, criterion, optimizer, scaler, device, train=False)
        extra_acc = None
        if extra_val_loader:
            _, extra_acc = run_epoch(model, extra_val_loader, criterion, optimizer, scaler, device, train=False)
        return fer_acc, extra_acc

    best_val_acc, epochs_without_improvement, history = 0.0, 0, []
    # Resume an interrupted run of this same candidate; otherwise start from
    # --init-from or ImageNet. Re-evaluate so a worse epoch never overwrites it.
    start = model_path if os.path.exists(model_path) else args.init_from
    if start:
        print(f"[INFO] Starting from {start}")
        model.load_state_dict(torch.load(start, map_location=device, weights_only=True))
        if start == model_path:
            best_val_acc = select_val_acc(*evaluate())
            print(f"[INFO] Resumed val_acc={best_val_acc:.4f}")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        fer_acc, extra_acc = evaluate()
        val_acc = select_val_acc(fer_acc, extra_acc)
        scheduler.step()
        epoch_time = time.time() - t0
        print(f"[EPOCH {epoch}/{args.max_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"fer_val={fer_acc:.4f} extra_val={extra_acc} ({epoch_time:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "fer_val_acc": fer_acc, "extra_val_acc": extra_acc, "epoch_seconds": epoch_time})
        if val_acc > best_val_acc:
            best_val_acc, epochs_without_improvement = val_acc, 0
            torch.save(model.state_dict(), model_path)
            print(f"[OK] New best val_acc={val_acc:.4f} -- saved to {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[INFO] Early stopping -- no improvement in {PATIENCE} epochs")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_loss, test_acc = run_epoch(model, fer_test_loader, criterion, optimizer, scaler, device, train=False)
    print(f"[FINAL] fer_test_acc={test_acc:.4f} (labels={args.labels}; not comparable across label sets)")
    report = {"args": vars(args), "labels": FER7_LABELS, "img_size": IMG_SIZE,
              "best_val_acc": best_val_acc, "fer_test_acc": test_acc, "fer_test_loss": test_loss,
              "epochs_trained": len(history), "history": history}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Report saved to {report_path}")
```

Actualizar también el docstring del módulo, en la sección `Usage:`:

```
Usage:
    python -m backend.ml.train_emotion_cnn --name <candidato> [--labels ferplus]
        [--extra-manifest data/training/cremad/manifest.csv] [--init-from X.pt]
        [--sad-weight-mult 1.5] [--num-workers 0] [--batch-size 32]
Writes backend/ml/saved_models/candidatos/<candidato>.pt -- never emotion_cnn.pt.
```

- [ ] **Step 4: Run tests**

Run: `pytest backend/tests/test_train_helpers.py -v` → 9 PASS. Luego `pytest backend/tests/` → todo PASS.

- [ ] **Step 5: Humo de extremo a extremo (1 época)**

Run: `python -m backend.ml.train_emotion_cnn --name humo --labels ferplus --max-epochs 1`
Expected: aparece `[EPOCH 1/1]`, se crean `saved_models/candidatos/humo.pt` y `humo.json`, y `emotion_cnn.pt` conserva la misma fecha de modificación. Anotar `epoch_seconds` para estimar las tandas. Después, borrar `humo.pt` y `humo.json`.

- [ ] **Step 6: Commit**

```bash
git add backend/ml/train_emotion_cnn.py backend/tests/test_train_helpers.py
git commit -m "Train into named candidates, with FER+ labels and an extra face manifest"
```

---

### Task 6: Tandas de entrenamiento, evaluación y promoción

Protocolo fijado **antes** de ver resultados: tres tandas declaradas, las tres se evalúan y no se añaden tandas nuevas a la vista de los números. Si ninguna pasa, se registra el resultado negativo, como en `adf36ee`.

**Antes de empezar (pasos diferidos).** Estas comprobaciones con datos reales se aplazaron porque Smart App Control bloqueó `_ssl.pyd` en el entorno cuando se escribió el código (sin HTTPS no había descargas). Hacerlas en este orden, antes del Step 1:

1. `python -m backend.ml.ferplus_labels --verificar` (Task 3, Step 5): debe acabar en `[OK] FER+ join verified`.
2. Humo de CREMA-D (Task 4, Step 5): `--max-actors 2`, inspeccionar los PNG (incluido el primer fotograma muestreado de varios clips) y ejecutar el comprobador de fugas de una línea.
3. Humo de entrenamiento (Task 5, Step 5): 1 época con `--name humo`, y después borrar `humo.pt` y `humo.json`.

**Files:**
- Create: `backend/ml/resultados/{ferplus,ferplus_cremad,ferplus_cremad_sad15}.json`
- Modify: `backend/app/references.py` (entrada "Alcance real del reconocimiento de emoción", ~línea 539)
- Modify (solo si hay promoción): `backend/ml/saved_models/emotion_cnn.pt` (no versionado)

- [ ] **Step 1: Extracción completa de CREMA-D** (en primer plano; horas, porque la descarga domina; ~6 GB en tránsito)

Run: `python -m backend.ml.cremad_faces --delete-videos`
Expected: `actors train=61 val=10 excluded=20`, y al final un manifiesto con unas 25–29k filas. Si se corta, relanzar el mismo comando: continúa por donde iba.

Comprobar también el reparto con `python -c "import csv,collections;r=list(csv.DictReader(open('data/training/cremad/manifest.csv')));print(len(r),collections.Counter(x['split'] for x in r))"` y que el `[OK]` final no reporte muchas descargas fallidas; si faltan muchas filas (cuota LFS de GitHub agotada), relanzar más tarde: continúa donde iba.

- [ ] **Step 2: Tanda A — solo FER+**

Run: `python -m backend.ml.train_emotion_cnn --name ferplus --labels ferplus --num-workers 2`

- [ ] **Step 3: Tanda B — FER+ y CREMA-D, partiendo de A**

Run: `python -m backend.ml.train_emotion_cnn --name ferplus_cremad --labels ferplus --extra-manifest data/training/cremad/manifest.csv --init-from backend/ml/saved_models/candidatos/ferplus.pt --num-workers 2`

- [ ] **Step 4: Tanda C — B con `sad` sobreponderado**

Run: `python -m backend.ml.train_emotion_cnn --name ferplus_cremad_sad15 --labels ferplus --extra-manifest data/training/cremad/manifest.csv --init-from backend/ml/saved_models/candidatos/ferplus.pt --sad-weight-mult 1.5 --num-workers 2`

- [ ] **Step 5: Evaluar las tres en el arnés CREMA-D**

Para cada `X` en `ferplus`, `ferplus_cremad` y `ferplus_cremad_sad15`, ejecutar:

```bash
python data/test_videos/evaluate_cremad.py --modelo backend/ml/saved_models/candidatos/X.pt --salida backend/ml/resultados/X.json
python -m backend.ml.eval_summary backend/ml/resultados/baseline_fer2013.json backend/ml/resultados/X.json
```

Expected: cada comparación imprime los deltas y `PROMOCIONABLE` o `NO PROMOCIONABLE` con los motivos. Elegir, entre las que dicen `PROMOCIONABLE`, la de mayor `global`.

- [ ] **Step 6: Comprobación end to end en otro estudio** (solo si hay candidato elegido)

Hacer copia de seguridad y copiar el candidato:

```bash
cp backend/ml/saved_models/emotion_cnn.pt backend/ml/saved_models/emotion_cnn_fer2013_baseline.pt
cp backend/ml/saved_models/candidatos/<elegido>.pt backend/ml/saved_models/emotion_cnn.pt
```

Arrancar el servidor (`python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`) y, en otra terminal, ejecutar `python data/test_videos/evaluate.py`.
Expected: global en RAVDESS actor 1 ≥ 46 % y `sad` > 1/8. Si baja del 46 %, **revertir** (`cp emotion_cnn_fer2013_baseline.pt emotion_cnn.pt`) y tratarlo como no promocionable: el candidato solo aprendió el estudio de CREMA-D.

- [ ] **Step 7: Tests**

Run: `pytest backend/tests/` → todo PASS.

- [ ] **Step 8: Registrar en `references.py`**

Actualizar `current_value` y `note` de la entrada "Alcance real del reconocimiento de emoción" con los números reales de `resultados/*.json`: global, `sad`, peor y mejor actor, brecha por sexo y el resultado en RAVDESS. Indicar qué candidato se promovió o que no se promovió ninguno, y con qué datos se entrenó: FER+ (etiquetas MS) y CREMA-D FaceVote, con 61 actores de entrenamiento, 10 de validación y los 20 de evaluación excluidos.

- [ ] **Step 9: Commit**

```bash
git add backend/ml/resultados/ backend/app/references.py
git commit -m "Record the FER+ and CREMA-D retraining results against the cross-actor harness"
```

---

## Después de este plan (fuera de alcance, por orden)

1. Si el sesgo por actor o por sexo sigue sin moverse: **RAF-DB** (pedir la EULA por email) como tanda D, con el mismo protocolo.
2. Con el disco y la RAM nuevos: **AffectNet** (acuerdo académico). En una GTX 1650, una época sobre ~280k imágenes tarda horas, así que conviene antes medir si la vía A1 (pesos ya publicados, entrenados en AffectNet) basta.
3. Umbral de decisión por clase para `sad` (idea 3 de la nota): se prueba encima del modelo que quede, sin reentrenar.
