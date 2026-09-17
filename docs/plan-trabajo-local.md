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
