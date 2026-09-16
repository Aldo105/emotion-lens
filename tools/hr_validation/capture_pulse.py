"""
Graba las lecturas del sensor de pulso con marca de tiempo absoluta.

Se ejecuta en paralelo a una sesion normal de EmotionLens. No se comunica con
la app: cada lectura se guarda con time.time(), y compare_hr.py las alinea
despues usando la hora de creacion de la sesion. Acoplar los dos procesos
solo anadiria formas de que la prueba falle.

Uso:
    python capture_pulse.py                  # autodetecta el puerto
    python capture_pulse.py --port COM5
    python capture_pulse.py --list           # lista puertos disponibles
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("Falta pyserial. Instalalo con:  pip install pyserial")

AQUI = Path(__file__).parent


def listar_puertos():
    puertos = list(list_ports.comports())
    if not puertos:
        print("No se detecto ningun puerto serie.")
        return
    print("Puertos disponibles:")
    for p in puertos:
        print(f"  {p.device:<8} {p.description}")


def autodetectar():
    """Busca un puerto que parezca un Arduino."""
    for p in list_ports.comports():
        desc = (p.description or "").lower()
        fabricante = (p.manufacturer or "").lower()
        if any(k in desc or k in fabricante
               for k in ("arduino", "ch340", "ch910", "wch", "usb-serial", "cp210")):
            return p.device
    puertos = list(list_ports.comports())
    return puertos[0].device if len(puertos) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="puerto serie, p.ej. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--list", action="store_true", help="listar puertos y salir")
    ap.add_argument("--out", help="archivo de salida")
    args = ap.parse_args()

    if args.list:
        listar_puertos()
        return

    puerto = args.port or autodetectar()
    if not puerto:
        print("No pude determinar el puerto. Usa --list para verlos y --port para elegir.")
        return

    salida = Path(args.out) if args.out else (
        AQUI / f"pulso_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    print(f"puerto : {puerto} @ {args.baud}")
    print(f"salida : {salida.name}")
    print("\nColoca el dedo sobre el sensor y arranca la sesion en EmotionLens.")
    print("Ctrl+C para terminar.\n")

    n = 0
    try:
        with serial.Serial(puerto, args.baud, timeout=2) as ser, \
                open(salida, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tiempo_unix", "millis_arduino", "bpm", "bpm_promedio"])

            while True:
                linea = ser.readline().decode("utf-8", errors="replace").strip()
                if not linea or linea.startswith("#"):
                    if linea:
                        print(f"  {linea}")
                    continue

                partes = linea.split(",")
                if len(partes) < 3:
                    continue
                try:
                    millis, bpm, promedio = int(partes[0]), float(partes[1]), float(partes[2])
                except ValueError:
                    continue

                w.writerow([f"{time.time():.3f}", millis, bpm, promedio])
                f.flush()   # por si se corta la corriente a mitad de sesion
                n += 1
                if n % 10 == 0:
                    print(f"  {n} latidos | actual {bpm:.0f} bpm | promedio {promedio:.0f} bpm")

    except KeyboardInterrupt:
        print(f"\nterminado: {n} latidos guardados en {salida.name}")
    except serial.SerialException as e:
        print(f"Error de puerto serie: {e}")


if __name__ == "__main__":
    main()
