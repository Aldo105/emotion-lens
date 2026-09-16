"""
Construye un video de prueba con emociones etiquetadas a partir de RAVDESS.

RAVDESS codifica la etiqueta en el nombre del archivo:
    modalidad-canal-EMOCION-intensidad-frase-repeticion-actor.mp4
El tercer campo es la emocion. Los clips duran 3-5s, pero el pipeline exige
30s de calibracion, asi que el video empieza con un tramo neutral largo y
despues encadena los clips de emocion, guardando el instante de cada uno
como ground truth.

Uso:
    python build_test_video.py
"""
import json
import os
import sys
from pathlib import Path

import cv2

AQUI = Path(__file__).parent
SALIDA_VIDEO = AQUI / "emotion_test.mp4"
SALIDA_LABELS = AQUI / "emotion_test_labels.json"

# Tercer campo del nombre -> etiqueta del sistema
RAVDESS_EMO = {
    "01": "neutral",
    "02": "neutral",    # 'calm' no tiene equivalente propio en el sistema
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

CALIB_SECONDS = 35      # margen sobre los 30s que exige el pipeline
FPS_SALIDA = 30.0


def clips_disponibles(raiz: Path):
    encontrados = []
    for p in raiz.rglob("*.mp4"):
        partes = p.stem.split("-")
        if len(partes) < 7:
            continue
        # El dataset trae cada toma dos veces: 01 = audio+video, 02 = solo
        # video. Nos quedamos con una para no duplicar cada clip.
        if partes[0] != "01":
            continue
        emo = RAVDESS_EMO.get(partes[2])
        if not emo:
            continue
        encontrados.append({
            "path": p,
            "emocion": emo,
            "codigo": partes[2],
            "intensidad": "fuerte" if partes[3] == "02" else "normal",
        })
    return encontrados


def escribir_clip(writer, path, size):
    """Escribe los frames de un clip, reescalando. Devuelve cuantos escribio."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
        n += 1
    cap.release()
    return n


def main():
    clips = clips_disponibles(AQUI)
    if not clips:
        sys.exit("No se encontraron clips .mp4 de RAVDESS en esta carpeta")

    print(f"clips encontrados: {len(clips)}")
    por_emocion = {}
    for c in clips:
        por_emocion.setdefault(c["emocion"], []).append(c)
    for emo in sorted(por_emocion):
        print(f"  {emo:<10} {len(por_emocion[emo])}")

    neutrales = por_emocion.get("neutral", [])
    if not neutrales:
        sys.exit("Sin clips neutrales: no se puede construir el tramo de calibracion")

    # Resolucion y fps del primer clip
    cap = cv2.VideoCapture(str(clips[0]["path"]))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or FPS_SALIDA
    cap.release()
    size = (w, h)
    print(f"\nresolucion: {w}x{h} @ {fps_src:.1f} fps")

    writer = None
    for codec in ["mp4v", "avc1"]:
        writer = cv2.VideoWriter(str(SALIDA_VIDEO), cv2.VideoWriter_fourcc(*codec),
                                 fps_src, size)
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        sys.exit("No se pudo abrir el VideoWriter")

    total_frames = 0
    etiquetas = []

    # ── Tramo de calibracion: clips neutrales repetidos ──────────────
    print(f"\nescribiendo tramo de calibracion ({CALIB_SECONDS}s)...")
    objetivo = int(CALIB_SECONDS * fps_src)
    i = 0
    while total_frames < objetivo:
        clip = neutrales[i % len(neutrales)]
        total_frames += escribir_clip(writer, clip["path"], size)
        i += 1
    print(f"  {total_frames} frames ({total_frames/fps_src:.1f}s)")
    calib_fin = total_frames / fps_src

    # ── Clips de emocion, preferentemente de intensidad fuerte ───────
    orden = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    print("\nescribiendo clips de emocion...")
    for emo in orden:
        candidatos = por_emocion.get(emo, [])
        fuertes = [c for c in candidatos if c["intensidad"] == "fuerte"] or candidatos
        for clip in fuertes[:2]:      # dos por emocion
            inicio = total_frames / fps_src
            n = escribir_clip(writer, clip["path"], size)
            total_frames += n
            fin = total_frames / fps_src
            etiquetas.append({
                "emocion": emo,
                "inicio": round(inicio, 2),
                "fin": round(fin, 2),
                "intensidad": clip["intensidad"],
                "archivo": clip["path"].name,
            })
            print(f"  {emo:<10} {inicio:6.1f}s - {fin:6.1f}s  ({clip['path'].name})")

    writer.release()

    meta = {
        "video": SALIDA_VIDEO.name,
        "fps": fps_src,
        "duracion_total": round(total_frames / fps_src, 2),
        "calibracion_hasta": round(calib_fin, 2),
        "fuente": "RAVDESS (Livingstone & Russo, 2018) — CC BY-NC-SA 4.0",
        "segmentos": etiquetas,
    }
    SALIDA_LABELS.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    mb = SALIDA_VIDEO.stat().st_size / 1024 / 1024
    print(f"\nvideo   : {SALIDA_VIDEO.name} ({mb:.1f} MB, {meta['duracion_total']:.1f}s)")
    print(f"etiquetas: {SALIDA_LABELS.name} ({len(etiquetas)} segmentos)")


if __name__ == "__main__":
    main()
