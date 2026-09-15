# Plan: migrar EmotionLens de prueba y error a evidencia científica

Objetivo: que ninguna constante que gobierne el comportamiento del sistema sea un
número elegido a ojo. Cada una debe quedar en uno de tres estados —
**VALIDADO** (valor publicado), **CALIBRADO** (ajustado contra un dataset con
codificación FACS certificada) o **HEURÍSTICO** (sin respaldo, y mostrado como tal
al usuario).

Estado de partida: **13 de 17 grupos de constantes carecen de respaldo publicado.**
El inventario completo está en `backend/app/references.py` y se publica en
`docs/bibliografia-cientifica.pdf`.

**Estado tras la Fase 1: 7 VALIDADO / 0 CALIBRADO / 13 HEURÍSTICO (20 grupos —**
**se añadieron 3 al documentar el patrón 'stress' y dos bugs de ritmo cardíaco**
**encontrados en el camino).**

---

## Antes de la sesión de código: solicitar los datasets

La Fase 3 no puede empezar sin datos con codificación FACS hecha por humanos
certificados. Los tres datasets relevantes se distribuyen bajo acuerdo académico y
la aprobación tarda **días o semanas**. Conviene solicitarlos ya, aunque el código
se escriba después.

| Dataset | Qué aporta | Institución |
|---|---|---|
| **DISFA** | Intensidad AU 0–5, frame a frame, 27 sujetos — es el que permite calibrar la geometría de AUs | University of Denver (Mohammad Mahoor) |
| **CASME II** | 247 micro-expresiones a 200 fps con etiquetas AU — valida la capa temporal | Chinese Academy of Sciences (Xiaolan Fu) |
| **SAMM** | 159 micro-movimientos a 200 fps, 7 categorías — conjunto de prueba independiente | Manchester Metropolitan University (Moi Hoon Yap) |

DISFA es el más importante: sin intensidades AU etiquetadas no hay forma de fijar
las ~30 constantes de `action_units.py`.

---

## Fase 0 — Infraestructura de procedencia ✅ hecha

- `backend/app/references.py` — registro de bibliografía + inventario de constantes.
- `backend/scripts/generate_bibliography.py` — genera el PDF desde el registro.
- `docs/bibliografia-cientifica.pdf` — 15 referencias verificadas con DOI.

El PDF se genera **desde el código**, así que no puede desincronizarse. Regla que
adoptamos: ninguna constante nueva entra sin su entrada en el registro.

```bash
python -m backend.scripts.generate_bibliography
```

---

## Fase 1 — Correcciones directas contra la literatura ✓ hecha

No requiere datasets. Son casos donde existe un valor publicado y el código usa otro.

### 1.1 Umbral de parpadeo (EAR) ✓
`action_units.py:88` usaba `0.15`; Soukupová & Čech (2016) publican `0.20`. Además la
fórmula EAR estándar promedia **dos** pares verticales de landmarks y el código usaba
uno solo. **Corregido:** umbral a 0.20 y EAR ahora promedia dos pares por ojo.

### 1.2 Ancla de la tasa de parpadeo ← **corrige un sesgo real** ✓
`action_units.py:466` normalizaba con `(bpm - 15) / 25`, es decir contra la tasa de
**reposo**. Bentivoglio et al. (1997) reportan ~17 ppm en reposo pero **~26 ppm en
conversación**. Una entrevista es una conversación: un candidato que parpadea con
total normalidad se registraba como "elevado", lo que bajaba su score fisiológico y
por tanto su congruencia. **Corregido:** ancla reubicada a 26 ppm (norma
conversacional) como respaldo, reemplazada por el basal propio del sujeto en cuanto
la calibración de 30s lo calcula — la normalización intra-sujeto que el rango
individual amplio (4–48 ppm) exige.

### 1.3 Prototipos AU→emoción incompletos ✓
`MICRO_EXPR_PATTERNS` (`micro_expressions.py:54-97`) no coincidía con los prototipos
EMFACS publicados:

| Emoción | EMFACS | Antes | Faltaba | Estado |
|---|---|---|---|---|
| Miedo | 1+2+4+5+7+20+25/26 | req 1,4 / sup 2,20,25 | AU5 | ✓ agregado a sup |
| Ira | 4+5+7+23 | req 4,7 / sup 23,24 | AU5 | ✓ agregado a sup |
| Sorpresa | 1+2+5+26 | req 2,25 / sup 26 | AU1, AU5 | ✓ agregados a sup |
| Tristeza | 1+4+15 | req 1,15 / sup 17 | AU4 | ✓ agregado a sup |
| Asco | 9+15+16 (+6,11,17) | req 9 / sup 15,25 | AU25 sobraba | ✓ quitado |
| Desprecio | 12+14 unilateral | igual | — | sin cambios |

La categoría `"stress"` (AU23+AU24) **no es un prototipo EMFACS**: es invención del
proyecto. Se mantuvo (código externo depende de la etiqueta) pero **se marcó
explícitamente como heurística** en el propio código y en `references.py`.

### 1.4 Implementar AU5 (Upper Lid Raiser) ✓
`action_units.py` calculaba 16 AUs pero no AU5, que aparece en 3 de los 6
prototipos de emoción básica. Su ausencia sesgaba miedo, ira y sorpresa hacia falsos
negativos. **Implementado** geométricamente desde la malla (apertura del párpado
superior respecto al iris, landmarks 468/473); el umbral numérico queda marcado
HEURÍSTICO (pendiente de calibración contra DISFA, Fase 3), igual que las demás
constantes geométricas de AUs.

### 1.5 Ventana de duración modal ✓
La bonificación de "duración óptima" usaba 100–250 ms; Yan et al. (2013) sitúan la
moda en **80–200 ms**. **Corregido.**

### 1.6 Banda rPPG ✓
`0.8–2.0 Hz` (48–120 BPM) era correcta pero estrecha para entrevistas con estrés
situacional. **Ampliada** a `0.7–3.0 Hz` (42–180 BPM), lo habitual en la literatura
rPPG (`config.py: evm_freq_low/high`).

### Bugs encontrados durante la Fase 1 (no estaban en el plan original)

Al diagnosticar por qué el ritmo cardíaco devolvía BPM = 0, aparecieron dos bugs de
implementación — no de respaldo científico — en `heart_rate.py`:

- **Resta del ROI de referencia:** se restaba el valor RGB crudo del puente nasal al
  de la frente para cancelar ruido de iluminación. Como la nariz suele ser más
  brillante, esto producía medias negativas que disparaban la guarda de
  `_compute_chrom_signal` y dejaban el BPM fijo en 0. **Corregido:** ahora se resta
  solo la *deriva* del ROI de referencia respecto a su propio basal (EMA), preservando
  el nivel absoluto de brillo que CHROM necesita.
- **Umbral de movimiento demasiado sensible:** a 640×480, el umbral de 5.0 equivalía
  a solo ~4 px de desplazamiento — por debajo del jitter propio de MediaPipe
  (~2–5 px) — así que casi todos los frames se descartaban y el buffer nunca se
  llenaba. **Corregido:** subido a 15.0.

Ambos quedan documentados como HEURÍSTICO en `references.py` (no existe un valor
publicado para ninguno de los dos).

---

## Fase 2 — Reencuadre de afirmaciones

Aquí no se cambian números sino lo que el sistema **dice** sobre una persona. Tres
resultados de la literatura acotan lo defendible:

- **Bond & DePaulo (2006)**: sobre 24.483 jueces, la precisión humana para detectar
  mentiras es 54%. Los meta-análisis no hallan relación fiable entre señales
  faciales y veracidad.
- **Barrett et al. (2019)**: la correspondencia entre configuración facial y estado
  interno es altamente variable entre personas, contextos y culturas.
- **NRC (2003)**: la premisa "desviarse del basal indica ocultamiento" es la del
  polígrafo (CQT), evaluada como de base científica débil y tasa de error desconocida.

Acciones:

1. Renombrar el concepto de **"Trustworthiness / Confiabilidad"** a algo descriptivo
   y defendible: *índice de variabilidad expresiva*. El sistema mide variabilidad,
   no honestidad.
2. Quitar de `congruence.py:224-248` la inferencia de "possible concealment". Medir
   y reportar la desviación como dato descriptivo, sin interpretarla.
3. Revisar `interview_analyzer.py`: `"social_masking"` y `red_flags` afirman más de
   lo que la evidencia permite. Reformular como observaciones, no como banderas.
4. Que el reporte PDF marque visualmente cada métrica con su nivel de evidencia, para
   que un entrevistador no confunda un compuesto heurístico con una medición.

---

## Fase 3 — Calibración empírica (requiere los datasets y tu GPU)

### 3.1 Geometría de AUs contra DISFA
Las ~30 constantes de `action_units.py` (`(ratio - 0.12) * 8.0` y similares) no son
ciencia: son factores de ajuste de la malla de MediaPipe. Procedimiento:

1. Correr el analizador sobre los videos de DISFA.
2. Para cada AU, regresión entre el valor geométrico crudo y la intensidad 0–5
   codificada por humanos.
3. Sustituir cada constante por los coeficientes ajustados.
4. Reportar correlación (ICC / Pearson) por AU — y marcar como **HEURÍSTICO** todo
   AU cuya correlación quede baja, en vez de fingir que está calibrado.

Esto convierte ~30 constantes de HEURÍSTICO a CALIBRADO, con error medido.

### 3.2 Capa temporal contra CASME II, prueba en SAMM
Calibrar umbrales de detección (`onset_ratio > 2.0`, `threshold + var * 2.5`,
`min_detection_gap`) maximizando F1 contra las micro-expresiones etiquetadas de
CASME II. Evaluar en SAMM **sin reajustar**, para tener una cifra honesta de
generalización.

### 3.3 El score de congruencia — decisión de producto pendiente

Los pesos `0.30/0.35/0.20/0.15` y los cortes `80/50` **no pueden validarse contra la
literatura**, porque no existe ningún dataset etiquetado con "congruencia verdadera".
No es que falte buscar la cita: la cita no puede existir. Hay tres salidas honestas,
y hay que elegir una:

| Opción | Qué implica |
|---|---|
| **A. Derivar contra un proxy medible** | Ajustar los pesos para predecir algo que sí se puede medir (p. ej. ansiedad auto-reportada con STAI antes/después). Requiere recolectar datos con consentimiento. Es lo más riguroso. |
| **B. Declarar heurístico y hacerlo ajustable** | Mantener el score, exponer los pesos en la UI y etiquetarlo visiblemente como heurístico. Rápido y honesto, pero el número sigue sin significar nada verificable. |
| **C. Eliminar el compuesto** | Mostrar los 4 componentes por separado sin fusionarlos. Se pierde el "gauge" del dashboard, se gana que todo lo mostrado es medible. |

Mi recomendación es **C para el reporte y B para el dashboard en vivo**: el número
único es útil como señal de atención durante la entrevista, pero no debería quedar
impreso en un documento que alguien use para decidir una contratación.

---

## Fase 4 — Validación

- Tests de regresión que fijen los valores calibrados y fallen si alguien los toca
  sin actualizar el registro de procedencia.
- Test que verifique que toda constante en `config.py` tiene entrada en
  `references.py` — así la regla se hace cumplir sola.
- Reportar en el README las métricas reales de acuerdo con ground truth por AU, en
  lugar de la tabla de efectividad actual, que no está medida.

---

## Fase 5 — Regenerar documentación

Volver a correr el generador (el PDF refleja automáticamente los nuevos niveles) y
actualizar el README con: qué está calibrado, contra qué dataset, con qué error, y
qué sigue siendo heurístico.

---

## Orden sugerido en la sesión

1. Fase 1 completa — son correcciones directas, bajo riesgo, alto valor (1.2 corrige
   un sesgo que hoy perjudica a candidatos normales).
2. Fase 2 — decidir el reencuadre contigo, porque cambia qué es el producto.
3. Fase 3 sólo cuando lleguen los datasets.

Las Fases 1 y 2 no dependen de nada externo: se pueden hacer completas en una sesión.
