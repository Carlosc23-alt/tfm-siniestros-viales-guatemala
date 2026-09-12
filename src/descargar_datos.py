"""
descargar_datos.py
===================

Automatiza la descarga de los microdatos oficiales de siniestralidad vial del INE/PNC
(Capitulo 2.1 y 2.2 de la memoria), para que el TFM sea reproducible sin depender de
que alguien haya guardado copias locales de los .xlsx, y para poder refrescar los datos
cuando el INE publique un nuevo anio (ver reentrenar_modelo.py y Capitulo 7.3 de la memoria).

Estrategia de dos niveles:
  1. Intenta autodescubrir los recursos vigentes via la API generica de CKAN
     (la plataforma que usa datos.ine.gob.gt). Esto es lo correcto para que el script
     siga funcionando cuando el INE publique 2025, 2026, etc. sin tener que tocar codigo.
  2. Si la API no responde (cambio de plataforma, bloqueo de red, mantenimiento), cae a un
     diccionario de URLs conocidas, capturadas en agosto de 2026 al escribir este TFM.

Verificacion (30 de agosto de 2026): se ejecuto este script contra el portal real y se
compararon los 8 archivos obtenidos por la via automatica contra los que se habian descargado
a mano al inicio del trabajo. Los 8 resultaron identicos byte a byte (SHA-256), de modo que la
automatizacion reproduce exactamente el dataset con el que se entreno el modelo.

Uso:
    python3 src/descargar_datos.py             # descarga lo que falte en data/raw/
    python3 src/descargar_datos.py --forzar    # vuelve a descargar todo, aunque ya exista
    python3 src/descargar_datos.py --carpeta X # carpeta destino distinta
"""

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Este script requiere el paquete 'requests' (ver requirements.txt): pip install requests")
    sys.exit(1)

CKAN_API = "https://datos.ine.gob.gt/api/3/action/package_show"
DATASET_ID = "accidentes-de-transito-fallecidos-y-lesionados"
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_POR_DEFECTO = str(RAIZ / "data" / "raw")
MANIFIESTO_NOMBRE = "manifiesto_descarga.json"

# URLs de respaldo, capturadas manualmente en agosto de 2026 desde
# https://datos.ine.gob.gt/en/dataset/accidentes-de-transito-fallecidos-y-lesionados
# Se usan solo si la API de CKAN no responde. Si el INE publica un anio nuevo y la API
# tampoco esta disponible, hay que agregar la URL aqui a mano.
URLS_RESPALDO = {
    "2018": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/fba61ef7-4958-4c71-b4bc-a54f9ab93a48/download/fallecidos-y-lesionados-ano-2018.xlsx",
    "2019": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/970198e6-83a9-4d65-bce8-5c905b2199a6/download/fallecidos-y-lesionados-ano-2019.xlsx",
    "2020": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/1994a185-3d73-45da-8f2e-d9d4cb906f07/download/fallecidos-y-lesionados-ano-2020.xlsx",
    "2021": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/114f228e-2e9e-4465-9108-5d11b834c655/download/fallecidos-y-lesionados-ano-2021.xlsx",
    "2022": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/53221342-ad93-42a5-b1bd-3b930686d68d/download/fallecidos-y-lesionados-ano-2022.xlsx",
    "2023": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/7fe294c2-b120-44e7-a903-3250ae1f8462/download/fallecidos-y-lesionados-2023-pnc.xlsx",
    "2024": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/d4505add-cbc1-4a1a-a35b-dcd73694d0fe/download/base-de-datos-fallecidos-y-lesionados-pnc-2024.xlsx",
    "diccionario": "https://datos.ine.gob.gt/dataset/b80e7836-a158-4830-9177-bca5a3b6cf0d/resource/6e9838fd-cc71-491c-8f53-8292fac6649b/download/diccionario-fallecidos-y-lesionados.xlsx",
}

CABECERAS = {
    "User-Agent": "TFM-UCM-CarlosCalderon/1.0 (uso academico; datos abiertos INE Guatemala)"
}


def obtener_recursos_via_api(timeout=20):
    """Intenta descubrir los recursos vigentes via la API generica de CKAN.

    Devuelve un dict {clave: url}. La clave es '2024', '2025', ... o 'diccionario',
    extraida por expresion regular del nombre del recurso que reporta CKAN.
    Lanza una excepcion si la API no responde o el formato no es el esperado.
    """
    resp = requests.get(CKAN_API, params={"id": DATASET_ID}, timeout=timeout, headers=CABECERAS)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError("La API de CKAN respondio pero success=false")

    recursos = {}
    for recurso in payload["result"]["resources"]:
        nombre = (recurso.get("name") or "").lower()
        url = recurso.get("url")
        if not url:
            continue
        if "diccionario" in nombre:
            recursos["diccionario"] = url
            continue
        anios = re.findall(r"(20\d{2})", nombre)
        if anios:
            recursos[anios[-1]] = url  # si el nombre trae dos numeros de 4 digitos, se asume que el ultimo es el anio
    if not recursos:
        raise RuntimeError("La API respondio pero no se pudo identificar ningun recurso por anio")
    return recursos


def descargar_archivo(url, destino, timeout=60):
    """Descarga un archivo a `destino` de forma atomica (escribe a .tmp y luego renombra)."""
    tmp = destino + ".tmp"
    with requests.get(url, stream=True, timeout=timeout, headers=CABECERAS) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, destino)


def nombre_archivo_local(clave, url):
    if clave == "diccionario":
        return "diccionario-fallecidos-y-lesionados.xlsx"
    extension = os.path.splitext(url)[1] or ".xlsx"
    return f"fallecidos_lesionados_{clave}{extension}"


def descargar_todos(carpeta=CARPETA_POR_DEFECTO, forzar=False, timeout=60):
    os.makedirs(carpeta, exist_ok=True)

    try:
        recursos = obtener_recursos_via_api(timeout=min(timeout, 20))
        fuente = "API CKAN (autodescubrimiento en tiempo real)"
    except Exception as exc:
        print(f"Aviso: no se pudo usar la API de CKAN ({exc}).")
        print("Usando las URLs de respaldo conocidas (capturadas en agosto de 2026).")
        recursos = URLS_RESPALDO
        fuente = "URLs de respaldo (agosto 2026)"

    print(f"Fuente de URLs: {fuente}")
    print(f"Recursos identificados: {sorted(recursos.keys())}\n")

    manifiesto = {
        "generado_en_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "fuente_urls": fuente,
        "archivos": [],
    }

    algun_error = False
    for clave, url in sorted(recursos.items()):
        destino = os.path.join(carpeta, nombre_archivo_local(clave, url))
        if os.path.exists(destino) and not forzar:
            print(f"  [{clave}] ya existe en {destino} -- se omite (usa --forzar para re-descargar)")
            manifiesto["archivos"].append({
                "clave": clave, "url": url, "ruta_local": destino, "estado": "ya_existia",
            })
            continue
        try:
            print(f"  [{clave}] descargando {url} ...")
            descargar_archivo(url, destino, timeout=timeout)
            tamano = os.path.getsize(destino)
            print(f"  [{clave}] OK -- {tamano / 1024:.0f} KB")
            manifiesto["archivos"].append({
                "clave": clave, "url": url, "ruta_local": destino,
                "estado": "descargado", "bytes": tamano,
            })
        except Exception as exc:
            algun_error = True
            print(f"  [{clave}] ERROR: {exc}")
            manifiesto["archivos"].append({
                "clave": clave, "url": url, "ruta_local": destino,
                "estado": "error", "detalle": str(exc),
            })

    ruta_manifiesto = os.path.join(carpeta, MANIFIESTO_NOMBRE)
    with open(ruta_manifiesto, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)
    print(f"\nManifiesto de descarga guardado en: {ruta_manifiesto}")

    if algun_error:
        print("\nHubo al menos un error de descarga -- revisa el manifiesto antes de reentrenar.")
    return manifiesto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", default=CARPETA_POR_DEFECTO,
                         help=f"Carpeta destino (por defecto: {CARPETA_POR_DEFECTO})")
    parser.add_argument("--forzar", action="store_true",
                         help="Re-descarga aunque el archivo ya exista localmente")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout por archivo, en segundos")
    args = parser.parse_args()

    manifiesto = descargar_todos(carpeta=args.carpeta, forzar=args.forzar, timeout=args.timeout)
    hubo_error = any(a["estado"] == "error" for a in manifiesto["archivos"])
    sys.exit(1 if hubo_error else 0)


if __name__ == "__main__":
    main()
