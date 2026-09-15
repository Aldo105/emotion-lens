# Plan: migrar EmotionLens de prueba y error a evidencia científica

Objetivo: que ninguna constante que gobierne el comportamiento del sistema sea un
número elegido a ojo. Cada una debe quedar en uno de tres estados —
**VALIDADO** (valor publicado), **CALIBRADO** (ajustado contra un dataset con
codificación FACS certificada) o **HEURÍSTICO** (sin respaldo, y mostrado como tal
al usuario).

Estado de partida: **13 de 17 grupos de constantes carecen de respaldo publicado.**
El inventario completo está en `backend/app/references.py` y se publica en
`docs/bibliografia-cientifica.pdf`.

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

## Fase 1 — Correcciones directas contra la literatura

No requiere datasets. Son casos donde existe un valor publicado y el código usa otro.

### 1.1 Umbral de parpadeo (EAR)
`action_units.py:88` usa `0.15`; Soukupová & Čech (2016) publican `0.20`. Además la
fórmula EAR estándar promedia **dos** pares verticales de landmarks y el código usa
uno solo. Corregir ambas cosas.

### 1.2 Ancla de la tasa de parpadeo ← **corrige un sesgo real**
`action_units.py:466` normaliza con `(bpm - 15) / 25`, es decir contra la tasa de
**reposo**. Bentivoglio et al. (1997) reportan ~17 ppm en reposo pero **~26 ppm en
conversación**. Una entrevista es una conversación: hoy un candidato que parpadea
con total normalidad se registra como "elevado", lo que baja su score fisiológico y
por tanto su congruencia. Reanclar a la norma conversacional.

Además, el rango individual sano es enorme (4–48 ppm), así que lo correcto no es un
corte absoluto sino normalizar contra el propio basal del sujeto medido durante la
calibración.

### 1.3 Prototipos AU→emoción incompletos
`MICRO_EXPR_PATTERNS` (`micro_expressions.py:54-90`) no coincide con los prototipos
EMFACS publicados:

| Emoción | EMFACS | En el código | Falta |
|---|---|---|---|
| Miedo | 1+2+4+5+7+20+25/26 | req 1,4 / sup 2,20,25 | **AU5** |
| Ira | 4+5+7+23 | req 4,7 / sup 23,24 | **AU5** |
| Sorpresa | 1+2+5+26 | req 2,25 / sup 26 | **AU1, AU5** |
| Tristeza | 1+4+15 | req 1,15 / sup 17 | **AU4** |
| Asco | 9+15+16 (+6,11,17) | req 9 / sup 15,25 | AU25 sobra |
| Desprecio | 12+14 unilateral | igual | — |

La categoría `"stress"` (AU23+AU24) **no es un prototipo EMFACS**: es invención del
proyecto. O se elimina, o se marca explícitamente como heurística.

### 1.4 Implementar AU5 (Upper Lid Raiser)
`action_units.py` calcula 16 AUs pero **no AU5**, que aparece en 3 de los 6
prototipos de emoción básica. Su ausencia sesga miedo, ira y sorpresa hacia falsos
negativos. Es geométricamente calculable desde la malla (apertura del párpado
superior respecto al iris).

### 1.5 Ventana de duración modal
La bonificación de "duración óptima" usa 100–250 ms; Yan et al. (2013) sitúan la
moda en **80–200 ms**. Ajuste menor, pero citable.

### 1.6 Banda rPPG
`0.8–2.0 Hz` (48–120 BPM) es correcta pero estrecha para entrevistas con estrés
situacional. Evaluar `0.7–3.0 Hz` (42–180 BPM), que es lo habitual en la literatura
rPPG.

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
