"""
Pasa el video etiquetado por la API real y compara contra el ground truth.

Sube el video como lo haria un usuario, espera a que termine el analisis y
lee la linea temporal desde el CSV exportado, de modo que mide el sistema
completo y no una version simplificada del pipeline.

Uso (con el servidor levantado en :8000):
    python evaluate.py
"""
import csv
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

AQUI = Path(__file__).parent
API = "http://127.0.0.1:8000/api"
VIDEO = AQUI / "emotion_test.mp4"
LABELS = AQUI / "emotion_test_labels.json"


def subir():
    with open(VIDEO, "rb") as f:
        r = requests.post(
            f"{API}/videos/upload",
            files={"file": (VIDEO.name, f, "video/mp4")},
            params={"session_name": "Prueba RAVDESS"},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()


def esperar(session_id, timeout=1800):
    t0 = time.time()
    ultimo = -1
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{API}/videos/{session_id}/progress", timeout=20)
            d = r.json()
        except Exception:
            time.sleep(3)
            continue

        pct = int((d.get("progress") or 0) * 100)
        if pct != ultimo:
            print(f"  {pct:3d}%  {d.get('status', '')}")
            ultimo = pct
        if d.get("status") in ("completed", "error"):
            return d.get("status")
        time.sleep(3)
    return "timeout"


def leer_timeline(session_id):
    r = requests.get(f"{API}/reports/{session_id}/csv", timeout=120)
    r.raise_for_status()
    filas = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    registros = []
    for fila in filas:
        ts = fila.get("timestamp") or fila.get("Timestamp")
        emo = fila.get("emotion") or fila.get("Emotion")
        if ts is None or emo is None:
            continue
        try:
            registros.append({"timestamp": float(ts), "emotion": emo.strip().lower()})
        except ValueError:
            continue
    return registros


def main():
    if not VIDEO.exists() or not LABELS.exists():
        sys.exit("Falta emotion_test.mp4 o sus etiquetas — corre build_test_video.py")

    meta = json.loads(LABELS.read_text(encoding="utf-8"))
    segmentos = meta["segmentos"]
    print(f"video: {meta['duracion_total']}s, {len(segmentos)} segmentos etiquetados")
    print(f"calibracion hasta: {meta['calibracion_hasta']}s\n")

    print("subiendo...")
    info = subir()
    sid = info.get("session_id") or info.get("id")
    print(f"session_id: {sid}\n")

    print("procesando:")
    estado = esperar(sid)
    print(f"\nestado final: {estado}")
    if estado != "completed":
        sys.exit("El analisis no termino correctamente")

    registros = leer_timeline(sid)
    print(f"frames en la linea temporal: {len(registros)}\n")
    if not registros:
        sys.exit("El CSV no trajo registros")

    aciertos = 0
    evaluables = 0
    confusion = defaultdict(Counter)
    por_intensidad = []

    print(f"{'':3}{'esperado':<10} {'detectado':<12} {'segmento':<16} reparto")
    print("-" * 78)
    for seg in segmentos:
        dentro = [r for r in registros if seg["inicio"] <= r["timestamp"] < seg["fin"]]
        if not dentro:
            print(f"   {seg['emocion']:<10} {'(sin datos)':<12} "
                  f"{seg['inicio']:>5.1f}-{seg['fin']:>5.1f}s")
            continue

        evaluables += 1
        cuenta = Counter(r["emotion"] for r in dentro)
        top = cuenta.most_common(1)[0][0]
        confusion[seg["emocion"]][top] += 1
        ok = top == seg["emocion"]
        aciertos += ok
        por_intensidad.append((seg.get("intensidad", "?"), ok))
        reparto = ", ".join(f"{e}:{n}" for e, n in cuenta.most_common(3))
        print(f"{'OK ' if ok else '   '}{seg['emocion']:<10} {top:<12} "
              f"{seg['inicio']:>5.1f}-{seg['fin']:>5.1f}s   {reparto}")

    print(f"\naciertos: {aciertos}/{evaluables} "
          f"({aciertos/max(evaluables,1)*100:.0f}%)")
    print("(al azar entre 9 etiquetas seria ~11%)")

    print("\npor emocion:")
    for esperado in sorted(confusion):
        total_emo = sum(confusion[esperado].values())
        ok_emo = confusion[esperado][esperado]
        barra = "#" * int(ok_emo / max(total_emo, 1) * 20)
        print(f"  {esperado:<10} {ok_emo:>2}/{total_emo:<2} "
              f"{ok_emo/max(total_emo,1)*100:>3.0f}%  {barra}")

    # Una expresion actuada en intensidad fuerte es el caso mas facil posible;
    # la intensidad normal se parece mas a lo que hace alguien en una entrevista.
    print("\npor intensidad:")
    for inten in ("fuerte", "normal"):
        sub = [s for s in por_intensidad if s[0] == inten]
        if sub:
            ok_i = sum(1 for _, o in sub if o)
            print(f"  {inten:<8} {ok_i:>2}/{len(sub):<2} "
                  f"({ok_i/len(sub)*100:>3.0f}%)")

    print("\nmatriz de confusion (esperado -> detectado):")
    for esperado in sorted(confusion):
        fila = ", ".join(f"{d}:{n}" for d, n in confusion[esperado].most_common())
        print(f"  {esperado:<10} -> {fila}")


if __name__ == "__main__":
    main()
