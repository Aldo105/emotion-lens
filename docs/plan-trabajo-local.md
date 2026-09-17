# Plan de trabajo — sesión local

Documento para retomar el proyecto desde una sesión de chat **en la computadora
propia**, donde sí hay cámara, GPU y permisos para ejecutar y probar. Contiene
el estado, el diagnóstico ya hecho y los pasos concretos a ejecutar.

Última actualización: 2026-09-17

---

## 0. Cómo arrancar la sesión local

```bash
conda activate emotion-lens
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000   ·   API docs: http://localhost:8000/docs
```

`backend/app/main.py:142-144` monta `frontend/` como estáticos en `/`, así que
no hace falta servidor aparte para la página.

Verificaciones rápidas antes de tocar nada:

```bash
curl http://localhost:8000/api/health
pytest backend/tests/ -q
python tools/hr_validation/simulate_estimator.py   # banco de pruebas rPPG
```

**Convención del proyecto:** todo valor numérico que se cambie en el pipeline
debe registrarse en `backend/app/references.py` como `ConstantEvidence` (con su
`tier`, referencia y nota). El repo trata la trazabilidad científica como parte
del código, no como documentación aparte.

---

## 1. PRIORIDAD ALTA — El ritmo cardíaco dejó de marcar

### 1.1 Síntoma reportado

Sesión 1: marcaba valores intermitentes, aparecían y desaparecían.
Sesiones 2, 3 y 4: se congeló y ya no marcó nada.

### 1.2 Origen

El último commit que tocó el estimador es `f95cee1` *"Stabilize live rPPG heart
rate estimation"* (2026-09-15). Añadió cuatro compuertas nuevas que deben
cumplirse **todas a la vez** para mostrar un número
(`backend/app/services/heart_rate.py:301-333`):

| Compuerta | Valor | Dónde |
|---|---|---|
| Ventana mínima de señal | 8 s | `heart_rate.py:288` |
| Confianza espectral mínima | 0.40 | `heart_rate.py:315` |
| Dispersión (IQR) de estimaciones | ≤ 12 BPM sobre 20 muestras | `heart_rate.py:349-364` |
| Frescura | última estimación válida < 5 s | `heart_rate.py:329-332` |

El propio commit y el registro en `references.py` declaran que esos umbrales se
calibraron **contra una señal sintética**, no contra grabaciones reales
("ajustada en simulación sintética, no contra ground truth humano"). Ese es el
origen del problema: son correctos en laboratorio y demasiado estrictos frente
a una webcam real.

### 1.3 Evidencia medida

Reproducible con `python tools/hr_validation/simulate_estimator.py`.

**(a) La compuerta de dispersión es un acantilado binario, no una degradación.**
Alimentando el estimador real con un pulso sintético de 72 BPM:

| amplitud | ruido | con lectura | BPM reportado |
|---|---|---|---|
| 0.40 | 0.30 | 93.3 % | 71.9 ✓ |
| 0.40 | 0.80 | **18.4 %** | 89.4 ✗ |
| 0.20 | 0.30 | **14.5 %** | 71.2 |
| 0.20 | 0.80 | **0.0 % (nunca)** | — |

Un cambio pequeño de relación señal/ruido pasa de "funciona el 93 % del tiempo"
a "no marca nunca". Una webcam real cruza ese umbral constantemente según la
luz, la compresión JPEG y el movimiento — exactamente el patrón "a veces sí, a
veces no, y luego nunca".

**(b) Una vez contaminado el historial, tarda 7-11 s de señal *perfecta* en
volver.** `_raw_history` es un buffer fijo de 20 estimaciones que solo avanza
con estimaciones aceptadas; mientras conserve valores dispersos de una mala
racha, el IQR sigue por encima de 12 BPM. En una entrevista real, donde la
persona habla y se mueve, casi nunca hay 10 s limpios seguidos → se queda en
blanco de forma permanente.

**(c) Perder frames sesga el BPM hacia abajo de forma sistemática.** Con un
pulso **perfecto** de 72 BPM y sin ruido, solo interrumpiendo el muestreo:

| frames perdidos | BPM reportado |
|---|---|
| 0 % | 72.5 ✓ |
| 10 % | 65.7 |
| 20 % | 54.6 |
| 50 % | 50.1 |

Causa: `heart_rate.py:739-757` calcula `actual_fps = (n-1)/span` y luego aplica
la FFT **como si el muestreo fuera uniforme**. Cada frame perdido estira el eje
temporal efectivo y baja la frecuencia detectada proporcionalmente. Esto
invalida cualquier medición, incluso cuando sí muestra un número.

**(d) El pipeline le corta la señal al estimador.** En
`backend/app/routers/websocket.py:556-562`, el *quality gate* hace `continue`
cuando la calidad de cámara puntúa < 0.4 — y ese `continue` está **antes** del
paso de ritmo cardíaco (`websocket.py:667-673`). Consecuencias:

- El estimador no recibe esos frames: el buffer deja de llenarse y a los 5 s la
  lectura caduca.
- No se envía `frame_result`, así que `updateHeartRate` ni siquiera se ejecuta
  (`frontend/js/dashboard.js:318-328`) y el panel **se queda congelado con el
  último número en pantalla**.
- La calidad se recalcula solo cada 10 frames (`websocket.py:550`), así que cada
  fallo arrastra ~10 frames seguidos.

Las penalizaciones de calidad se acumulan fácil: rostro pequeño (−0.15), imagen
poco nítida (−0.30), poco contraste (−0.10), cabeza girada (−0.08), luz lateral
(−0.15), poca luz (−0.10) suman 0.88 → puntaje 0.12 = `fail`. Además la nitidez
se mide sobre el frame **preprocesado** (CLAHE + gamma + denoise), y el denoise
baja la varianza del Laplaciano, es decir, el propio preprocesado empuja el
puntaje hacia `fail`.

**(e) No se está guardando nada.** `websocket.py:676` condiciona la persistencia
de `hr_bpm` a `signal_ready`. Si nunca es `True`, ninguna sesión guarda ritmo
cardíaco, y `tools/hr_validation/compare_hr.py` se queda sin datos que comparar
contra el sensor de contacto.

### 1.4 Plan de arreglo (en este orden)

**Paso 1 — Que el quality gate deje de matar la señal.** *(el más barato, y
explica el congelamiento)*
En `websocket.py`, mover el paso 5.5 de ritmo cardíaco **antes** del `continue`
del quality gate, o no aplicar ese `continue` al rPPG. El estimador solo necesita
el ROI de la frente: la nitidez, el encuadre y el giro de cabeza penalizan la
clasificación de emociones, no la fotopletismografía. Enviar siempre
`frame_result` con `heart_rate`, aunque la calidad sea baja.

**Paso 2 — Corregir el muestreo no uniforme.** *(invalida las mediciones)*
En `_estimate_heart_rate()` y `_compute_signal_quality()`, interpolar la señal a
una rejilla temporal uniforme con `np.interp` sobre `self._timestamps` antes de
la FFT, en vez de asumir espaciado constante. Alternativa más rigurosa:
`scipy.signal.lombscargle`, diseñado para muestreo irregular. Verificar con la
prueba `huecos`: el BPM reportado debe quedarse en ~72 con 20 % de pérdida.

**Paso 3 — Guardar siempre la lectura.** Quitar la condición `signal_ready` de la
persistencia en `websocket.py:676-684` y guardar `bpm`, `bpm_confidence` y el
motivo de bloqueo aunque no se muestre. Sin esto no se puede validar nada contra
el sensor.

**Paso 4 — Instrumentar el motivo del bloqueo.** Añadir al resultado de
`process_frame` un campo `block_reason` (`warming_up`, `low_confidence`,
`inconsistent_peak`, `stale`, `no_signal`) y mostrarlo en el panel. Hoy un
"--" no distingue entre "calentando", "señal débil" y "el pipeline no me manda
frames", que es justo lo que hizo lento este diagnóstico.

**Paso 5 — Reajustar las compuertas con datos reales.** Solo después de los
pasos anteriores, y midiendo contra el sensor de contacto:
- Acotar la dispersión por **ventana de tiempo** (p. ej. últimos 6 s) en vez de
  por número fijo de estimaciones, para que una mala racha caduque sola.
- Vaciar `_raw_history` y `_hr_history` cuando la señal lleva perdida más de
  `stale_after_seconds`, para que la recuperación empiece limpia.
- Considerar mostrar el valor con un distintivo de confianza en vez de ocultarlo
  por completo — degradar suave en lugar de binario.
- Registrar cada umbral nuevo en `references.py`.

**Paso 6 — Validar.** Grabar 2-3 sesiones con el pulsómetro de
`tools/hr_validation/` y correr `compare_hr.py`. La meta del README es MAE de
3-6 BPM en condiciones de oficina; sin esta comparación, cualquier ajuste vuelve
a ser calibración sintética.

---

## 1b. Botón de microexpresiones del reporte final

### Síntoma

En el reporte final terminado, el botón "Video Microexpresiones" no funciona,
mientras que el de "Video EVM" sí.

### Causa (ya corregida en el repo)

`micro_expression_highlight_renderer.py` ubicaba cada clip con
`center_frame = int(round(timestamp * fps))`, usando los 20 FPS **nominales**.
El MP4 se declara a 20 FPS, pero el navegador entrega frames con un
`setInterval` best-effort (`frontend/js/config.js:12`) y la tasa real es menor
y variable, así que el desfase **crece con la duración de la sesión**: para una
sesión de 3 minutos entregada a 14 FPS, un evento en t=150 s se buscaba en el
frame 3000 de un video que solo tiene ~2520. El clip se descartaba por
"demasiado corto", y si se descartan todos, `_render_impl` devuelve `None`, el
estado queda en `failed` y el botón solo muestra el aviso de error.

Por eso fallaba solo este botón: el renderer EVM recorre el video en bloques
secuenciales y nunca convierte un tiempo a un índice de frame.

El grabador **ya guardaba** el timestamp real de cada frame
(`video_recorder.py:136`), pero ningún consumidor lo leía. El arreglo usa ese
arreglo con `np.searchsorted` (`_clip_bounds`), con pruebas en
`backend/tests/test_micro_highlight_bounds.py`.

**Falta verificarlo con una sesión real**: grabar una sesión con al menos una
microexpresión tardía y comprobar que el video se genera y que cada clip cae
sobre el gesto correcto.

### Pendientes relacionados (no corregidos)

1. **Las sesiones de video subido nunca generan este video.**
   `video_processor.py` no graba video crudo ni lanza el render, así que para
   esas sesiones el botón siempre dirá "no disponible". Decidir si se soporta
   o si el botón se oculta cuando `input_type != "webcam"`.
2. **La banda del filtro del renderer usa la FPS nominal.** `nyquist = fps / 2`
   con `fps=20` mientras la tasa real es ~14 desplaza la banda de 2-12 Hz de la
   amplificación. Conviene derivar la FPS efectiva de los timestamps del clip.
3. **El overlay de BPM del video EVM tiene el mismo defecto** (`evm_renderer.py`
   línea ~500, `current_time = frame_idx / fps`): no rompe nada, pero etiqueta
   cada lectura con un tiempo desplazado.

---

## 1c. Calibración guiada por pose de cabeza

### Qué se añadió

La calibración ya no son 30 s mirando al frente, sino una secuencia guiada de
cinco poses: centro, derecha, izquierda, abajo, arriba. La pantalla indica qué
pose sostener y solo cuenta muestras cuando el ángulo medido coincide con el
pedido, así que una pose que no se sostiene no se da por calibrada.

**Motivo:** las Action Units se calculan sobre distancias entre landmarks 2D, y
girar la cabeza las acorta por escorzo. Con una sola línea base frontal,
cualquier frame con la cabeza girada aparece como desviación respecto al reposo
aunque la cara no haya cambiado de expresión — infla el detector de
microexpresiones y sesga al clasificador justo cuando la persona deja de mirar
a la cámara, que en una entrevista es la mayor parte del tiempo.

**Cómo se aplica:** `PoseBaselines.to_frontal()` resta de cada frame la
diferencia entre la línea base interpolada para el ángulo actual y la del
centro, antes de que el resto del pipeline lo vea. Los consumidores
(microexpresiones, congruencia, clasificador) siguen trabajando contra la línea
base frontal y no necesitaron cambios.

Se interpola entre poses vecinas con Shepard (IDW) en vez de elegir la más
cercana: un salto de línea base al cruzar de una región a otra se detectaría
aguas abajo como una microexpresión falsa. Fuera de la región calibrada la
corrección se desvanece en vez de extrapolar.

Archivos: `backend/app/services/pose_calibration.py` (nuevo, 17 pruebas en
`backend/tests/test_pose_calibration.py`), integración en `websocket.py`,
instrucciones en `dashboard.js`, constantes registradas en `references.py`.

### Falta verificar con webcam real

1. **El signo del yaw.** `_estimate_head_pose` deriva el ángulo del
   desplazamiento de la nariz; no verifiqué contra cámara si yaw positivo
   corresponde a "derecha" tal como se le pide al sujeto. Si las instrucciones
   salen invertidas, se intercambian los valores de `right` y `left` en
   `POSE_ANCHORS`.
2. **Que los ángulos pedidos sean alcanzables** y que MediaPipe mantenga los
   landmarks en ellos. Si a ±22° pierde precisión, bajar las anclas.
3. **Que la corrección reduzca de verdad los falsos positivos.** Medir la tasa
   de microexpresiones detectadas con la cabeza girada, con y sin corrección:
   es el número que justifica toda la función.
4. **Duración total.** Con los conteos actuales son ~12 s de centro y ~6 s por
   giro de tiempo sostenido, más lo que tarde la persona en colocarse. Si
   resulta pesado, bajar `pose_calibration_turned_samples`.
5. **El *quality gate* puede colgar la calibración.** El `continue` de
   `websocket.py:556-562` descarta el frame antes del paso de calibración, igual
   que hace con el ritmo cardíaco (ver 1.3d). Si la cámara puntúa `fail` de
   forma sostenida, la secuencia no avanza *y el temporizador de pose tampoco
   corre*, porque solo se evalúa en frames que pasaron el gate: se queda
   esperando sin aviso. El arreglo del Paso 1 de la sección 1.4 (mover el gate
   o no aplicarlo a estas etapas) cubre también este caso; mientras tanto, es
   la primera sospecha si la calibración se queda parada en un paso.
6. **Menos datos para movimientos habituales.** El rastreador de gestos
   habituales ahora recibe solo los frames frontales alineados (~240 en vez de
   ~600). Si aparecen microexpresiones falsas por tics no filtrados, subir
   `pose_calibration_center_samples`.

### Botón de recalibrar

Estaba condicionado a `baseline_calibrated`: si la primera calibración no había
terminado, el clic no hacía absolutamente nada y sin aviso alguno — justo
cuando una sesión va mal y el sujeto recurre al botón. Ya no depende de eso, y
reinicia la secuencia de poses completa.

---

## 2. PRIORIDAD MEDIA — Despliegue permanente (VPS económico, CPU)

Archivos ya preparados en `deploy/` (ver `docs/deployment.md`, sección
"Budget CPU VPS"). Pasos pendientes, que requieren cuenta y tarjeta:

1. Contratar VPS Ubuntu 22.04, 2 vCPU / 4 GB (Hetzner ~9 USD/mes,
   DigitalOcean ~24 USD/mes) y anotar la IP.
2. Registrar dominio y apuntar un registro A a esa IP. HTTPS es obligatorio: el
   navegador bloquea `getUserMedia` (cámara) en orígenes sin TLS.
3. En el VPS: `sudo bash deploy/setup_vps.sh tu-dominio.com <url-del-repo>`
   (instala dependencias CPU, PostgreSQL, systemd con `Restart=always`, Nginx,
   Certbot y firewall).
4. Verificar `https://tu-dominio.com` y `/api/health`.
5. Para actualizar después: `sudo -u emotionlens bash deploy/deploy.sh`.
6. Opcional: configurar los secrets `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY` en
   GitHub para que `.github/workflows/deploy.yml` despliegue solo.

Limitación aceptada: sin GPU, la inferencia CNN en vivo procesa menos frames por
segundo que la máquina local. Es el camino barato para tener URL pública; migrar
a instancia con GPU si el rendimiento con varios usuarios lo exige.

---

## 3. Notas para la sesión local

- Probar el ritmo cardíaco **con cámara real** después de cada paso: el banco
  sintético confirma la lógica, pero el problema original apareció solo con
  webcam.
- Sentarse quieto y con buena luz frontal para la primera prueba, para separar
  "el arreglo funciona" de "las condiciones eran malas".
- Mirar la consola del navegador y los logs de uvicorn en paralelo: el mensaje
  `quality_warning` indica cuántos frames se están descartando.
