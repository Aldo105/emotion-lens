"""
Descarga un subconjunto diverso de CREMA-D.

CREMA-D tiene 91 actores con demografía declarada, frente al único actor de
RAVDESS. Eso permite dos cosas que hasta ahora no se podían medir:

  1. Si lo calibrado generaliza a otras caras, o está ajustado a una persona.
  2. Si funciona igual para todos los grupos demográficos — un sesgo conocido
     en reconocimiento facial, y especialmente grave en una herramienta que
     interviene en contrataciones.

Solo se baja la frase IEO, que es la única con los tres niveles de intensidad
(LO/MD/HI): ~33 MB en vez de los 7.5 GB del dataset completo.

Licencia: Open Database License (ODbL). Los archivos quedan fuera del repo.

Uso:
    python download_cremad.py            # 12 actores
    python download_cremad.py --actores 20
"""
import argparse
import csv
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).parent
DESTINO = AQUI / "cremad"
BASE = ("https://media.githubusercontent.com/media/"
        "CheyneyComputerScience/CREMA-D/master/VideoFlash")
DEMOG = AQUI / "crema_demographics.csv"

# CREMA-D no incluye 'surprise'
EMO = {"ANG": "angry", "DIS": "disgust", "FEA": "fear",
       "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
INTENS = {"LO": "baja", "MD": "media", "HI": "alta", "XX": "sin especificar"}


def elegir_actores(n):
    """Reparte la selección entre sexos, etnias y edades."""
    if not DEMOG.exists():
        sys.exit(f"Falta {DEMOG.name}. Descargalo de VideoDemographics.csv del repo.")

    with open(DEMOG, encoding="utf-8-sig") as f:
        actores = list(csv.DictReader(f))

    grupos = defaultdict(list)
    for a in actores:
        grupos[(a["Sex"], a["Race"])].append(a)

    # Dentro de cada grupo se alternan extremos de edad. Tomar los primeros de
    # una lista ordenada daba una selección entera de 20 a 33 años, que es
    # precisamente lo que impediría detectar un sesgo por edad.
    for clave, g in grupos.items():
        g.sort(key=lambda a: int(a["Age"]))
        intercalado = []
        i, j = 0, len(g) - 1
        while i <= j:
            intercalado.append(g[i])
            if i != j:
                intercalado.append(g[j])
            i += 1
            j -= 1
        grupos[clave] = intercalado

    elegidos, i = [], 0
    while len(elegidos) < n:
        avance = False
        for clave in sorted(grupos):
            if i < len(grupos[clave]) and len(elegidos) < n:
                elegidos.append(grupos[clave][i])
                avance = True
        if not avance:
            break
        i += 1
    return elegidos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actores", type=int, default=12)
    args = ap.parse_args()

    elegidos = elegir_actores(args.actores)
    DESTINO.mkdir(exist_ok=True)

    print(f"{len(elegidos)} actores seleccionados:")
    for a in elegidos:
        print(f"  {a['ActorID']}  {a['Age']:>2} años  {a['Sex']:<7} {a['Race']}")

    # Por actor: 5 emociones en intensidad baja y alta, mas el neutral
    objetivos = []
    for a in elegidos:
        aid = a["ActorID"]
        for cod in ("ANG", "DIS", "FEA", "HAP", "SAD"):
            for inten in ("LO", "HI"):
                objetivos.append((aid, f"{aid}_IEO_{cod}_{inten}.flv", EMO[cod], INTENS[inten]))
        objetivos.append((aid, f"{aid}_IEO_NEU_XX.flv", "neutral", INTENS["XX"]))

    print(f"\ndescargando {len(objetivos)} clips (~{len(objetivos)*0.25:.0f} MB)...")
    indice, fallos = [], 0
    for i, (aid, nombre, emocion, intensidad) in enumerate(objetivos, 1):
        destino = DESTINO / nombre
        if not destino.exists():
            try:
                urllib.request.urlretrieve(f"{BASE}/{nombre}", destino)
            except Exception as e:
                fallos += 1
                print(f"  fallo {nombre}: {e}")
                continue
        # Un puntero LFS pesa unos cientos de bytes; un video real, cientos de KB
        if destino.stat().st_size < 10_000:
            destino.unlink()
            fallos += 1
            continue

        demo = next(a for a in elegidos if a["ActorID"] == aid)
        indice.append({
            "archivo": nombre,
            "actor": aid,
            "emocion": emocion,
            "intensidad": intensidad,
            "edad": int(demo["Age"]),
            "sexo": demo["Sex"],
            "etnia": demo["Race"],
        })
        if i % 20 == 0:
            print(f"  {i}/{len(objetivos)}")

    (DESTINO / "indice.json").write_text(
        json.dumps({"fuente": "CREMA-D (Cao et al., 2014) — ODbL",
                    "clips": indice}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    mb = sum(f.stat().st_size for f in DESTINO.glob("*.flv")) / 1024 / 1024
    print(f"\n{len(indice)} clips descargados ({mb:.1f} MB), {fallos} fallos")
    print(f"indice: {DESTINO.name}/indice.json")


if __name__ == "__main__":
    main()
