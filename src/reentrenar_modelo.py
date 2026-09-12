"""
reentrenar_modelo.py
=====================

Orquesta el reentrenamiento del modelo cuando el INE publique datos nuevos (Capitulo 7.3 de la
memoria, "Model Monitoring" / reentrenamiento del CRISP-DM). Reusa el notebook consolidado
(TFM_Analisis_Completo.ipynb) como "job" de entrenamiento en vez de duplicar el pipeline de ETL +
feature engineering + modelado en un segundo archivo -- así no hay dos copias de la misma lógica
que puedan desincronizarse con el tiempo.

Pasos:
  1. (opcional, activado por defecto) Descarga los datos mas recientes del INE con
     descargar_datos.py.
  2. Hace una copia de respaldo de los artefactos actualmente desplegados en modelo/
     (el modelo "candidato anterior").
  3. Ejecuta el notebook completo de punta a punta con `jupyter nbconvert --execute`, que
     regenera modelo/ con un modelo recien entrenado y sus metricas en
     modelo/metricas_modelo_tfm.json.
  4. Compara las metricas del modelo nuevo contra las del respaldo (ROC-AUC en el holdout
     Out-of-Time como criterio principal).
  5. Decide si promover el modelo nuevo a produccion:
       - Sin --promover: nunca reemplaza modelo/ en firme -- dejar el nuevo modelo en
         candidatos/ para revision humana y restaurar el respaldo. Este es el comportamiento
         por defecto.
       - Con --promover: reemplaza modelo/ SOLO si el ROC-AUC del modelo nuevo no es peor que
         el del modelo actual menos --tolerancia. Si es peor, se niega a promover salvo que se
         pase tambien --forzar.
  6. Registra cada corrida (metricas antes/despues, decision, promovido si/no) en
     reentrenamiento_log.jsonl, para trazabilidad -- el requisito de "Model Monitoring" que
     pedía la rúbrica del curso.

Estado de validacion: la orquestacion (respaldo / restauracion / comparacion / promocion / log)
esta probada contra un notebook de prueba que simula la ejecucion real. La ejecucion real de
notebooks/TFM_Analisis_Completo.ipynb via nbconvert -- que tarda bastante mas, por las dos
busquedas de hiperparametros, SHAP y Optuna -- todavia no se ha validado de punta a punta contra
un anio de datos realmente nuevo. Conviene hacerlo una vez antes de confiar en --promover sin
supervision (ver Capitulo 7.3 de la memoria).

Uso:
    python3 reentrenar_modelo.py                     # descarga, reentrena, compara, NO promueve
    python3 reentrenar_modelo.py --promover           # ademas promueve si no empeora
    python3 reentrenar_modelo.py --promover --forzar  # promueve aunque empeore (usar con cautela)
    python3 reentrenar_modelo.py --omitir-descarga    # usa los datos ya presentes localmente
    python3 reentrenar_modelo.py --tolerancia 0.02    # tolerancia de ROC-AUC (por defecto 0.01)
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Raiz del repositorio: este archivo vive en src/, de modo que la raiz es su carpeta padre.
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK = BASE_DIR / "notebooks" / "TFM_Analisis_Completo.ipynb"
CARPETA_MODELO = BASE_DIR / "modelo"
CARPETA_CANDIDATOS = BASE_DIR / "candidatos"
CARPETA_HISTORIAL = BASE_DIR / "modelo_historial"
ARCHIVO_METRICAS = "metricas_modelo_tfm.json"
LOG_PATH = BASE_DIR / "reentrenamiento_log.jsonl"

TOLERANCIA_ROC_AUC_POR_DEFECTO = 0.01
TIMEOUT_POR_CELDA_SEGUNDOS = 3600     # el notebook tiene celdas pesadas (busquedas de hiperparametros, SHAP)
TIMEOUT_TOTAL_SEGUNDOS = 3 * 60 * 60  # 3 horas de margen para toda la ejecucion


def log(mensaje):
    print(f"[reentrenar_modelo] {mensaje}")


def paso_1_descargar_datos(omitir):
    if omitir:
        log("Se omite la descarga de datos (--omitir-descarga) -- se usan los archivos locales existentes.")
        return
    log("Paso 1/5: descargando datos actualizados del INE...")
    try:
        sys.path.insert(0, str(BASE_DIR / "src"))
        import descargar_datos
        manifiesto = descargar_datos.descargar_todos(carpeta=str(BASE_DIR / "data" / "raw"))
        if any(a["estado"] == "error" for a in manifiesto["archivos"]):
            log("AVISO: al menos un archivo no se pudo descargar -- revisa el manifiesto antes de continuar.")
    except Exception as exc:
        log(f"AVISO: fallo la descarga automatica ({exc}). Se continua con los datos locales existentes, si hay.")


def paso_2_respaldar_modelo_actual():
    log("Paso 2/5: respaldando el modelo actualmente desplegado...")
    if not CARPETA_MODELO.exists():
        log("No hay un modelo/ previo desplegado (primera ejecucion) -- no hay nada que respaldar.")
        return None
    respaldo = BASE_DIR / "modelo_respaldo_temp"
    if respaldo.exists():
        shutil.rmtree(respaldo)
    shutil.copytree(CARPETA_MODELO, respaldo)
    log(f"Respaldo temporal creado en {respaldo}")
    return respaldo


def paso_3_ejecutar_notebook():
    log("Paso 3/5: ejecutando el notebook completo (esto puede tardar bastante; incluye dos "
        "busquedas de hiperparametros, SHAP y la verificacion con Optuna)...")
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"No se encontro el notebook en {NOTEBOOK}")

    salida = NOTEBOOK.with_name(NOTEBOOK.stem + "_reentrenado.ipynb")
    comando = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={TIMEOUT_POR_CELDA_SEGUNDOS}",
        "--output", salida.name,
        str(NOTEBOOK),
    ]
    log(f"Comando: {' '.join(comando)}")
    resultado = subprocess.run(
        comando, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=TIMEOUT_TOTAL_SEGUNDOS,
    )
    if resultado.returncode != 0:
        log("ERROR: la ejecucion del notebook fallo. Ultimas lineas de la salida de error:")
        print("\n".join(resultado.stderr.strip().splitlines()[-40:]))
        raise RuntimeError("Fallo la ejecucion de nbconvert --execute")
    log(f"Notebook ejecutado correctamente. Version ejecutada guardada en: {salida}")
    return salida


def paso_4_comparar_metricas(respaldo):
    log("Paso 4/5: comparando metricas del modelo nuevo contra el modelo anterior...")
    ruta_nuevas = CARPETA_MODELO / ARCHIVO_METRICAS
    if not ruta_nuevas.exists():
        raise FileNotFoundError(
            f"El notebook se ejecuto pero no genero {ruta_nuevas} -- revisa la seccion 9 del notebook."
        )
    with open(ruta_nuevas, encoding="utf-8") as f:
        metricas_nuevas = json.load(f)

    metricas_anteriores = None
    if respaldo is not None:
        ruta_anteriores = respaldo / ARCHIVO_METRICAS
        if ruta_anteriores.exists():
            with open(ruta_anteriores, encoding="utf-8") as f:
                metricas_anteriores = json.load(f)

    return metricas_nuevas, metricas_anteriores


def decidir_promocion(metricas_nuevas, metricas_anteriores, tolerancia, forzar):
    roc_nuevo = metricas_nuevas["metricas_holdout"]["roc_auc"]
    if metricas_anteriores is None:
        return True, f"No habia un modelo anterior desplegado -- se acepta el nuevo modelo (ROC-AUC {roc_nuevo:.4f})."

    roc_anterior = metricas_anteriores["metricas_holdout"]["roc_auc"]
    diferencia = roc_nuevo - roc_anterior
    if diferencia >= -tolerancia:
        return True, (f"ROC-AUC nuevo {roc_nuevo:.4f} vs. anterior {roc_anterior:.4f} "
                       f"(diferencia {diferencia:+.4f}, dentro de la tolerancia de {tolerancia}).")
    if forzar:
        return True, (f"ROC-AUC nuevo {roc_nuevo:.4f} es peor que el anterior {roc_anterior:.4f} "
                       f"(diferencia {diferencia:+.4f}), pero se promueve de todas formas por --forzar.")
    return False, (f"ROC-AUC nuevo {roc_nuevo:.4f} es peor que el anterior {roc_anterior:.4f} "
                    f"(diferencia {diferencia:+.4f}, fuera de la tolerancia de {tolerancia}). "
                    f"No se promueve. Usa --forzar si de verdad quieres desplegarlo de todas formas.")


def paso_5_promover_o_archivar(promover_solicitado, promover_aprobado, respaldo, metricas_nuevas):
    hoy = datetime.date.today().isoformat()
    if promover_solicitado and promover_aprobado:
        log("Paso 5/5: promoviendo el modelo nuevo a produccion...")
        if respaldo is not None:
            CARPETA_HISTORIAL.mkdir(exist_ok=True)
            destino_historial = CARPETA_HISTORIAL / f"modelo_{hoy}"
            if destino_historial.exists():
                shutil.rmtree(destino_historial)
            shutil.move(str(respaldo), str(destino_historial))
            log(f"Modelo anterior archivado en {destino_historial}")
        log(f"Modelo nuevo activo en {CARPETA_MODELO} (ya quedo ahi tras la ejecucion del notebook).")
        return True

    log("Paso 5/5: NO se promueve el modelo nuevo -- se restaura el modelo anterior en produccion.")
    CARPETA_CANDIDATOS.mkdir(exist_ok=True)
    destino_candidato = CARPETA_CANDIDATOS / f"candidato_{hoy}"
    if destino_candidato.exists():
        shutil.rmtree(destino_candidato)
    shutil.copytree(CARPETA_MODELO, destino_candidato)
    log(f"Modelo recien entrenado guardado para revision en {destino_candidato} (no esta en produccion).")

    if respaldo is not None:
        shutil.rmtree(CARPETA_MODELO)
        shutil.move(str(respaldo), str(CARPETA_MODELO))
        log("Modelo anterior restaurado en produccion.")
    return False


def registrar_log(metricas_nuevas, metricas_anteriores, promovido, motivo):
    entrada = {
        "fecha_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "metricas_nuevas": metricas_nuevas.get("metricas_holdout"),
        "metricas_anteriores": metricas_anteriores.get("metricas_holdout") if metricas_anteriores else None,
        "promovido": promovido,
        "motivo": motivo,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    log(f"Corrida registrada en {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omitir-descarga", action="store_true",
                         help="No descargar datos nuevos; usar los archivos locales existentes.")
    parser.add_argument("--promover", action="store_true",
                         help="Reemplazar el modelo en produccion si el nuevo no empeora (ver --tolerancia).")
    parser.add_argument("--forzar", action="store_true",
                         help="Junto con --promover: promueve aunque el nuevo modelo sea peor. Usar con cautela.")
    parser.add_argument("--tolerancia", type=float, default=TOLERANCIA_ROC_AUC_POR_DEFECTO,
                         help=f"Tolerancia de ROC-AUC para considerar 'no peor' (por defecto {TOLERANCIA_ROC_AUC_POR_DEFECTO}).")
    args = parser.parse_args()

    respaldo = None
    try:
        paso_1_descargar_datos(args.omitir_descarga)
        respaldo = paso_2_respaldar_modelo_actual()
        paso_3_ejecutar_notebook()
        metricas_nuevas, metricas_anteriores = paso_4_comparar_metricas(respaldo)

        promover_aprobado, motivo = decidir_promocion(
            metricas_nuevas, metricas_anteriores, args.tolerancia, args.forzar
        )
        log(motivo)

        promovido = paso_5_promover_o_archivar(args.promover, promover_aprobado, respaldo, metricas_nuevas)
        registrar_log(metricas_nuevas, metricas_anteriores, promovido, motivo)

        log("Listo." + (" Modelo promovido a produccion." if promovido else
                         " Modelo NO promovido -- revisa candidatos/ y decide manualmente."))
        sys.exit(0)
    except Exception as exc:
        log(f"ERROR: {exc}")
        if respaldo is not None and respaldo.exists() and not CARPETA_MODELO.exists():
            log("Restaurando el modelo anterior por seguridad tras el error...")
            shutil.move(str(respaldo), str(CARPETA_MODELO))
        sys.exit(1)


if __name__ == "__main__":
    main()
