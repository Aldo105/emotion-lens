# Guía Rápida — EmotionLens

## Iniciar
```
conda activate emotion-lens
python -m uvicorn backend.app.main:app --reload --port 8000
```
Abrir: http://localhost:8000

## Uso básico
1. **Iniciar Sesión** → acepta el aviso ético.
2. Calibración: ~30s con expresión neutral, mirando a cámara.
3. Dashboard en vivo: emoción, ritmo cardíaco, congruencia, microexpresiones.
4. Marca eventos con los botones (▶️ Inicio tarea, ⚠️ Error, 💬 Pregunta clave) o escribe notas.
5. **Finalizar Sesión** → encuesta opcional de validación.

## Consejos de cámara
Sigue las sugerencias 💡 en pantalla (luz, distancia, postura) — mejoran el accuracy.

## Ver resultados
- Pestaña **Sesiones** → abrir cualquier sesión → PDF / CSV / video EVM.
- **Comparar**: dos sesiones lado a lado.
- **Subir Video**: analiza una grabación existente.

## Sin cámara a mano
```
python -m backend.scripts.seed_demo_session
```
Crea una sesión de ejemplo completa para explorar la app.
