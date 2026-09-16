"""
Compara el ritmo cardiaco estimado por camara (rPPG) contra el sensor de contacto.

Las dos fuentes se graban por separado y se alinean por hora absoluta: el CSV
del sensor guarda time.time() por latido, y la sesion de la app tiene su
created_at mas el timestamp relativo de cada registro.

Uso:
    python compare_hr.py pulso_20260916_101500.csv            # ultima sesion
    python compare_hr.py pulso_....csv --session 12
    python compare_hr.py pulso_....csv --offset -1.5          # ajuste manual
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Falta requests. Instalalo con:  pip install requests")

from datetime import datetime, timezone

API = "http://127.0.0.1:8000/api"
TOLERANCIA = 3.0     # segundos para emparejar una lectura con otra


def leer_sensor(path):
    filas = []
    with open(path, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            try:
                filas.append({
                    "t": float(fila["tiempo_unix"]),
                    "bpm": float(fila["bpm_promedio"] or fila["bpm"]),
                })
            except (ValueError, KeyError):
                continue
    return filas


def ultima_sesion():
    r = requests.get(f"{API}/sessions/", params={"limit": 1}, timeout=30)
    r.raise_for_status()
    s = r.json()["sessions"]
    return s[0] if s else None


def sesion_por_id(sid):
    r = requests.get(f"{API}/sessions/{sid}", timeout=30)
    r.raise_for_status()
    return r.json()


def leer_rppg(sid):
    """BPM por frame desde los action_units persistidos."""
    r = requests.get(f"{API}/reports/{sid}/data", timeout=120)
    r.raise_for_status()
    datos = r.json()
    registros = (datos.get("emotion_timeline") or datos.get("emotion_records") or [])
    salida = []
    for reg in registros:
        aus = reg.get("action_units") or {}
        bpm = aus.get("hr_bpm")
        if bpm:
            salida.append({
                "t_rel": float(reg["timestamp"]),
                "bpm": float(bpm),
                "conf": float(aus.get("hr_confidence") or 0.0),
            })
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sensor_csv")
    ap.add_argument("--session", type=int, help="id de sesion (por defecto, la ultima)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="segundos a sumar a los tiempos de la app si hay desfase")
    args = ap.parse_args()

    ruta = Path(args.sensor_csv)
    if not ruta.exists():
        sys.exit(f"No existe {ruta}")

    sensor = leer_sensor(ruta)
    if not sensor:
        sys.exit("El CSV del sensor no tiene lecturas validas")

    sesion = sesion_por_id(args.session) if args.session else ultima_sesion()
    if not sesion:
        sys.exit("No encontre la sesion")

    sid = sesion["id"]
    creada = datetime.fromisoformat(sesion["created_at"])
    if creada.tzinfo is None:
        creada = creada.replace(tzinfo=timezone.utc)
    origen = creada.timestamp() + args.offset

    rppg = leer_rppg(sid)
    print(f"sesion  : {sid} — {sesion.get('name')}")
    print(f"sensor  : {len(sensor)} latidos")
    print(f"rPPG    : {len(rppg)} lecturas con BPM")
    if not rppg:
        sys.exit("\nLa sesion no tiene BPM guardado. Necesita ser una sesion grabada\n"
                 "despues del cambio que persiste hr_bpm en los action_units.")

    # Emparejar cada lectura de la app con la del sensor mas cercana en el tiempo
    pares = []
    for r in rppg:
        t_abs = origen + r["t_rel"]
        cercano = min(sensor, key=lambda s: abs(s["t"] - t_abs))
        if abs(cercano["t"] - t_abs) <= TOLERANCIA:
            pares.append((t_abs, r["bpm"], cercano["bpm"], r["conf"]))

    if not pares:
        sys.exit("\nNinguna lectura coincidio en el tiempo. Revisa que ambas grabaciones\n"
                 "sean de la misma sesion, o prueba --offset para corregir el desfase.")

    errores = [abs(a - b) for _, a, b, _ in pares]
    rppg_v = [a for _, a, _, _ in pares]
    ref_v = [b for _, _, b, _ in pares]

    print(f"\nlecturas emparejadas: {len(pares)} (tolerancia {TOLERANCIA}s)\n")
    print(f"{'t(s)':>7} {'rPPG':>7} {'sensor':>8} {'error':>7} {'conf':>6}")
    print("-" * 40)
    paso = max(1, len(pares) // 15)
    for t_abs, a, b, c in pares[::paso]:
        print(f"{t_abs - origen:>7.1f} {a:>7.1f} {b:>8.1f} {a - b:>+7.1f} {c:>6.2f}")

    print(f"\nerror absoluto medio : {st.mean(errores):.1f} BPM")
    print(f"error mediano        : {st.median(errores):.1f} BPM")
    print(f"error maximo         : {max(errores):.1f} BPM")
    print(f"dentro de +-5 BPM    : {sum(e <= 5 for e in errores)}/{len(errores)}"
          f" ({sum(e <= 5 for e in errores)/len(errores)*100:.0f}%)")

    # La correlacion es lo que distingue una medicion de una constante con suerte
    if len(pares) > 2 and st.pstdev(rppg_v) > 0 and st.pstdev(ref_v) > 0:
        media_a, media_b = st.mean(rppg_v), st.mean(ref_v)
        cov = sum((a - media_a) * (b - media_b) for a, b in zip(rppg_v, ref_v)) / len(pares)
        r = cov / (st.pstdev(rppg_v) * st.pstdev(ref_v))
        print(f"correlacion          : {r:+.2f}")
        print(f"rango del sensor     : {min(ref_v):.0f}-{max(ref_v):.0f} BPM")
        if max(ref_v) - min(ref_v) < 10:
            print("  (ojo: el pulso apenas vario, la correlacion no es informativa —")
            print("   repite la prueba incluyendo un tramo con el pulso elevado)")
    else:
        print("correlacion          : no calculable")


if __name__ == "__main__":
    main()
