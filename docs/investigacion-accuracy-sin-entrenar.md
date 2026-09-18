# Cómo subir el accuracy sin entrenar un modelo

Investigación pendiente de ejecutar. Nada de esto está implementado.

Fecha: 2026-09-17

---

## 0. El punto de partida, con los números propios

Cualquier técnica se juzga contra lo que ya está medido en este repo
(`references.py`, "Alcance real del reconocimiento de emoción"):

| Medición | Valor |
|---|---|
| Accuracy sobre 20 actores (CREMA-D, 220 clips) | **36 %** |
| Accuracy con un solo actor (RAVDESS) | 46–54 % |
| Dispersión entre sujetos | **9 % (peor actor) → 73 % (mejor)** |
| `happy` | 95 % |
| `disgust` | 88 % → **15 %** al cambiar de cara |
| `angry` | 62 % → **22 %** al cambiar de cara |
| Mujeres vs hombres | 40 % vs 30 % |

Y lo ya descartado experimentalmente (commit `adf36ee`), que **no hay que
volver a intentar**:

- **Normalización por sujeto de la salida del CNN**: +2 puntos (34 → 36 %) y la
  dispersión entre actores se queda igual, en ~60 puntos.
- **Ruta de blendshapes**: 14 %.
- **Ruta de Action Units / prototipos EMFACS**: 3 %, por debajo del azar (17 %).
- **Resolución**: descartada. 480x360 da 64 % frente a 59 % a 1280x720.

La conclusión que quedó escrita fue *"the model has to be retrained"*.

**Esa conclusión es demasiado pesimista, y por una distinción concreta:**
*reentrenar* no es lo mismo que *reemplazar*. Usar pesos publicados por otra
persona no entrena nada. Ahí está la mayor parte del margen disponible.

---

## 1. El diagnóstico real: sesgo de identidad

La señal más informativa no es el 36 %, es el rango 9–73 %. Un modelo que
funciona con unas caras y falla con otras no tiene un problema de capacidad:
aprendió rasgos de identidad en vez de rasgos de expresión. Está documentado
como *identity bias*, y el efecto inter-racial llega a violar la invarianza de
medición ([PMC10733503](https://pmc.ncbi.nlm.nih.gov/articles/PMC10733503/)),
que es exactamente el modo de fallo que produce un rango de 60 puntos.

El clasificador actual arrastra esto desde su origen: **FER2013** son imágenes
de 48x48 en escala de grises, con etiquetas ruidosas y —dato que importa para
la sección 5— **caras sin registrar**, porque a esa resolución los detectores de
landmarks fallan ([arXiv 1612.02903](https://arxiv.org/pdf/1612.02903)). El
techo humano sobre FER2013 ronda el 65–68 %. Es el peor punto de partida
posible.

---

## 2. Nivel A — lo único que puede mover el 36 % de verdad

### A1. Cambiar los pesos por un modelo entrenado en AffectNet / RAF-DB

No se entrena nada: se descargan pesos publicados.

| Modelo | RAF-DB | FERPlus | AffectNet-7 |
|---|---|---|---|
| DDAMFN (backbone de CCFER) | 92.19 % | 91.24 % | **67.32 %** |
| POSTER++ | — | — | comparable |

AffectNet tiene ~450 k imágenes frente a las ~35 k de FER2013, en color, a
resolución útil y con muchísima más diversidad demográfica. Es la palanca
grande y no requiere entrenamiento.

Opciones concretas con licencia permisiva:

- [`ElenaRyumina/face_emotion_recognition`](https://huggingface.co/ElenaRyumina/face_emotion_recognition) — MIT.
- [`dwest1507/emotion-detection-model`](https://huggingface.co/dwest1507/emotion-detection-model) — MIT, exporta ONNX (~513 KB).
- Evitar [`DrGM/...ConvNeXt-V2L`](https://huggingface.co/DrGM/DrGM-ConvNeXt-V2L-Facial-Emotion-Recognition) — **CC-BY-NC-4.0, prohíbe uso comercial.**

> **Advertencia sobre estas cifras.** 67 % en AffectNet-7 es *test in-domain*,
> no es comparable con tu 36 % *cross-actor* sobre CREMA-D. No se debe creer
> ninguna mejora hasta medirla con tu propio `evaluate_cremad.py`. Ese script
> ya existe y es la prueba de aceptación correcta.

### A2. Modelos multi-tarea preentrenados

Savchenko publica modelos multi-tarea (MobileFaceNet / EfficientNet / DDAMFN)
que devuelven expresión **y** valencia/arousal a la vez
([CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/ABAW/papers/Savchenko_Leveraging_Pre-trained_Multi-task_Deep_Models_for_Trustworthy_Facial_Analysis_in_CVPRW_2024_paper.pdf)).
Encajan con la sección 4: dan la salida continua que este proyecto debería
estar reportando.

### A3. LibreFace

Toolkit de análisis facial de código abierto (WACV 2024,
[arXiv 2308.10713](https://arxiv.org/pdf/2308.10713)), con AUs e intensidades
preentrenadas. Sirve como segunda opinión independiente para contrastar contra
la ruta propia de AUs, que aquí mide 3 %.

---

## 3. Nivel B — ganancias incrementales, medidas y acumulables

Todas son de 1 a 3 puntos. **Ninguna arregla el 36 %**, pero son baratas y se
suman sobre el modelo que se acabe usando.

| Técnica | Ganancia publicada | Coste |
|---|---|---|
| **Test-time augmentation** (espejo horizontal + 7 vistas) | **+1.7 %** | N inferencias por frame |
| **Ensemble por voto blando** (7 modelos) | **+2.6 %** (73.2 → 75.8) | N modelos en memoria |
| **Agregación temporal** con codificador de secuencia | **+2.2 %** | bajo |
| **Test-time adaptation** (TENT / SAR, cross-dataset) | **+2.67 % / +2.61 %** | ver advertencia |

Fuentes: [arXiv 2004.11823](https://arxiv.org/pdf/2004.11823),
[arXiv 2603.19994](https://arxiv.org/html/2603.19994v1),
[arXiv 2403.11942](https://arxiv.org/html/2403.11942v1).

**B1. TTA con espejo horizontal.** Lo más barato del catálogo: la expresión
facial es casi simétrica, así que promediar la predicción de la imagen y su
espejo es gratis en términos de código. Cuesta el doble de inferencias por
frame, lo cual aquí importa porque el pipeline ya va justo (ver §4 del plan de
trabajo local). Aplicable solo cada N frames.

**B2. Ensemble.** Combinar 2–3 modelos publicados distintos por voto blando.
Se lleva bien con A1: el CNN actual puede quedarse como uno de los votantes.

**B3. Adaptación en tiempo de test — con una distinción importante.**
TENT y SAR minimizan entropía **con pasos de gradiente**: eso *es* entrenar,
aunque sea sobre el modelo ya hecho, y queda fuera de lo que pediste. Lo que sí
entra son los métodos **sin gradiente**, basados en caché o prototipos
([arXiv 2603.21309](https://arxiv.org/pdf/2603.21309)): guardan las predicciones
confiadas del sujeto actual y las reutilizan como referencia.

Esto encaja especialmente bien aquí, y merece atención: **una sesión de
EmotionLens es un único sujeto y ya tiene un periodo de calibración**. Es
justo el escenario para el que se diseñó la personalización en tiempo de test.
Ojo: la variante ingenua de esto (restar el perfil neutro) ya se midió y dio
+2 puntos. Los métodos de caché son más que eso, pero la expectativa debe ser
modesta.

**B4. Agregación temporal.** Hoy hay suavizado exponencial. Un decodificador de
secuencia (HMM con Viterbi sobre la serie de emociones) aprovecha que las
emociones tienen inercia y que ciertas transiciones son improbables, en vez de
promediar sin más.

---

## 4. Nivel C — no suben el número, pero mejoran lo que entregas

Esta sección es, en mi lectura, **la de mayor valor real** para este proyecto.

**C1. Abstención / predicción selectiva.** Con un rango de 9 % a 73 % según la
persona, para algunos sujetos la herramienta es ruido. Detectar *en qué sujeto
estás* y abstenerse vale más que +2 puntos de media. Un umbral de confianza
que diga "no puedo medir a esta persona de forma fiable" es honesto, es barato
y es defendible ante un comité de ética. La media sube sola porque solo
reportas donde aciertas.

**C2. Reducir el espacio de etiquetas.** Tus propios datos dicen que solo
`happy` sobrevive al cambio de cara (95 %), mientras `disgust` cae a 15 % y
`angry` a 22 %. Ya retiraste `fear` con ese mismo criterio. El paso siguiente
es evidente: reportar **positivo / neutro / negativo** en vez de 7 clases. Un
clasificador de 3 clases construido sobre las señales que sí generalizan será
mucho más preciso *y* mucho más defendible que uno de 7 que acierta una.

Esto es exactamente lo que argumenta Barrett et al. 2019, que ya está en tu
bibliografía: las categorías discretas no tienen la validez que se les supone.

**C3. Reportar valencia/arousal continuos.** Consecuencia de C2 y disponible
gratis en los modelos multi-tarea de A2.

---

## 5. Nivel D — preprocesado, con una trampa

**D1. Alineación facial.** La literatura dice que registrar la cara mejora
FER de forma significativa. **Pero hay una trampa específica de tu caso:**
FER2013 tiene las caras *sin registrar*, así que el modelo actual aprendió
sobre caras desalineadas. Alinear ahora en inferencia **crea un desajuste
train/test y puede empeorar las cosas**.

Conclusión: la alineación solo se paga **junto con** el cambio de modelo (A1),
porque los modelos modernos sí esperan entrada alineada. Hacerla sola es
arriesgado.

**D2. Normalización profunda de la cara (DeepFN).** Proyectar cada cara a una
plantilla común mejora la generalización de AUs entre identidades
([arXiv 2103.02484](https://arxiv.org/pdf/2103.02484)). Ataca directamente el
sesgo de identidad de la §1, que es *el* problema aquí.

---

## 6. Lo que ninguna de estas técnicas arregla

La diferencia de 40 % en mujeres frente a 30 % en hombres, y el efecto
inter-racial documentado, **no se corrigen sin tocar los datos de
entrenamiento**. Ninguna técnica de este documento lo hace.

Para una herramienta que interviene en contrataciones eso no es un detalle
técnico: es responsabilidad legal. La mitigación realista sin entrenar es C1
(abstenerse) más medir el rendimiento por grupo demográfico y publicarlo, que
es lo que ya hace `crema_demographics.csv`.

---

## 7. Orden sugerido

1. **A1 + medición con `evaluate_cremad.py`.** Es la única acción con margen
   para un cambio de magnitud, y el arnés de evaluación ya existe. Todo lo
   demás se decide después de ver ese número.
2. **C2** (colapsar a 3 clases) en paralelo: no depende del modelo y mejora la
   defensibilidad inmediatamente.
3. **C1** (abstención) — barato y es lo que convierte una media mala en una
   herramienta utilizable.
4. **B1 + B2** si hace falta exprimir puntos, midiendo el coste por frame.
5. **D1** solo si se hizo A1.
6. **B3** (caché sin gradiente) al final: encaja bien con el modelo de sesión,
   pero la variante simple ya dio solo +2.

## 8. Antes de creerse nada

Todo número citado aquí viene de *test in-domain* de otros. La única cifra
comparable con tu 36 % es la que salga de tu propio arnés cross-actor. Ninguna
de estas técnicas entra en producción sin pasar por
`data/test_videos/evaluate_cremad.py` y sin registrar el resultado en
`references.py`, igual que el resto de constantes del proyecto.
