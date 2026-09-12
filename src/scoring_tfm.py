"""
scoring_tfm.py -- Capa compartida de scoring del modelo del TFM.

Modelado Predictivo de la Severidad y Riesgo de Fatalidad en Siniestros Viales en Guatemala.
Carlos Alberto Calderon Illescas -- Master en Big Data, Data Science e IA (UCM).

Por que existe este modulo
--------------------------
La ingenieria de caracteristicas del Capitulo 3.3 tiene que aplicarse EXACTAMENTE igual en tres
lugares: al entrenar (notebook, seccion 5), al simular un escenario (app.py) y al puntuar un
reporte de campo (reporte_aseguradora.py). Mantener tres copias de esa logica hace que, en cuanto
el modelo gana una variable nueva, las copias se desincronicen en silencio y la prediccion falle
o -- peor -- se calcule sobre variables distintas a las del entrenamiento.

Este modulo centraliza esa logica en un solo lugar. Las apps lo importan; no reimplementan nada.
La lista de columnas que el modelo espera se lee de modelo/features_tfm.json, generada por la
seccion 9 del notebook, de modo que cualquier cambio en el conjunto de variables se propaga solo.

Uso
---
    from scoring_tfm import cargar_artefactos, construir_features, estimar_riesgo

    art = cargar_artefactos()
    X = construir_features(art, edad=22, sexo=1, departamento=1, municipio=101,
                           tipo_vehiculo=4, tipo_evento=1, marca_vehiculo="Toyota",
                           hora=3, mes=8, dia=15, dia_semana=6, anio=2026)
    resultado = estimar_riesgo(art, X)   # -> {"probabilidad": 0.37, "nivel": "Moderado", ...}
"""

import json
import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# La raiz del repositorio se resuelve desde la ubicacion de este archivo (src/), de modo que
# los scripts y las apps funcionan sin importar desde que carpeta se invoquen.
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_MODELO = str(RAIZ / "modelo")

# Umbrales de la banda de riesgo mostrada al usuario (Capitulo 6.2 de la memoria).
UMBRAL_RIESGO_BAJO = 0.20
UMBRAL_RIESGO_ALTO = 0.50

# Feriados oficiales fijos de Guatemala (mes, dia). Identico al notebook, seccion 5.
FERIADOS_FIJOS = {(1, 1), (5, 1), (9, 15), (10, 20), (11, 1), (12, 24), (12, 25), (12, 31)}


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------

def cargar_artefactos(carpeta=CARPETA_MODELO):
    """Carga los artefactos generados por la seccion 9 del notebook.

    Devuelve un dict con: preprocesador, modelo, frecuencias, diccionarios, features y metricas.
    Lanza FileNotFoundError con un mensaje accionable si falta alguno.
    """
    requeridos = {
        "preprocesador": "preprocesador_tfm.joblib",
        "modelo": "modelo_final_tfm.joblib",
        "frecuencias": "frecuencias_tfm.joblib",
        "diccionarios": "diccionarios_tfm.joblib",
    }
    faltantes = [n for n in requeridos.values() if not os.path.exists(os.path.join(carpeta, n))]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan artefactos del modelo en '{carpeta}/': {', '.join(faltantes)}. "
            "Ejecuta el notebook TFM_Analisis_Completo.ipynb completo (incluida la seccion 9)."
        )

    art = {clave: joblib.load(os.path.join(carpeta, arch)) for clave, arch in requeridos.items()}

    with open(os.path.join(carpeta, "features_tfm.json"), encoding="utf-8") as f:
        art["features"] = json.load(f)

    ruta_metricas = os.path.join(carpeta, "metricas_modelo_tfm.json")
    if os.path.exists(ruta_metricas):
        with open(ruta_metricas, encoding="utf-8") as f:
            art["metricas"] = json.load(f)
    else:
        art["metricas"] = {}

    return art


def columnas_esperadas(art):
    """Columnas exactas, y en el orden exacto, que el preprocesador espera recibir."""
    return art["features"]["num_features"] + art["features"]["cat_features"]


def resumen_metricas(art):
    """Texto corto con las metricas reales del modelo desplegado, para mostrarlo en la interfaz.

    Se lee del JSON de artefactos en vez de escribirlo a mano, para que nunca quede desfasado
    respecto al modelo que realmente esta cargado.
    """
    m = art.get("metricas", {}).get("metricas_holdout")
    if not m:
        return "Metricas del modelo no disponibles."
    return (
        f"ROC-AUC {m['roc_auc']:.3f} - Recall (fallecido) {m['recall_fallecido'] * 100:.1f}% "
        f"sobre el holdout {art['metricas'].get('anio_holdout', 2024)}"
    )


# ---------------------------------------------------------------------------
# Ingenieria de caracteristicas (replica exacta de la seccion 5 del notebook)
# ---------------------------------------------------------------------------

def calcular_domingo_pascua(anio):
    """Fecha del Domingo de Pascua (algoritmo de Meeus/Jones/Butcher). Cambia cada anio, por eso
    se calcula en vez de fijarse a mano."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return date(anio, mes, dia + 1)


def franja_horaria(hora):
    if pd.isna(hora):
        return "Desconocida"
    hora = int(hora)
    if hora < 6:
        return "Madrugada"
    if hora < 12:
        return "Manana"
    if hora < 19:
        return "Tarde"
    return "Noche"


def _feriado_y_temporada_alta(anio, mes, dia):
    """(es_feriado, es_temporada_alta_vial) a partir solo de anio/mes/dia, sin fuentes externas."""
    try:
        fecha = date(int(anio), int(mes), int(dia))
    except (ValueError, TypeError):
        return 0, 0
    pascua = calcular_domingo_pascua(fecha.year)
    en_semana_santa = (pascua - timedelta(days=3)) <= fecha <= pascua
    en_fin_de_anio = (fecha.month == 12 and fecha.day >= 20) or (fecha.month == 1 and fecha.day <= 2)
    es_feriado = int((fecha.month, fecha.day) in FERIADOS_FIJOS or en_semana_santa)
    es_temporada_alta = int(en_semana_santa or en_fin_de_anio)
    return es_feriado, es_temporada_alta


def construir_features(art, edad, sexo, departamento, municipio, tipo_vehiculo, tipo_evento,
                       marca_vehiculo, hora, mes, dia, dia_semana, anio=None):
    """Construye la fila de una sola observacion con exactamente las columnas del entrenamiento.

    `anio` solo interviene en el calculo de feriado / temporada alta vial; si no se indica se
    usa el anio en curso (el caso natural para un reporte de campo o una simulacion).
    """
    if anio is None:
        anio = date.today().year

    d = pd.DataFrame([{
        "edad": edad,
        "hora": hora,
        "mes": mes,
        "dia": dia,
        "sexo": sexo,
        "departamento": departamento,
        "tipo_vehiculo": tipo_vehiculo,
        "tipo_evento": tipo_evento,
        "municipio": municipio,
        "marca_vehiculo": marca_vehiculo,
        "dia_semana": dia_semana,
    }])

    vehiculos_vulnerables = art["diccionarios"]["VEHICULOS_VULNERABLES"]

    d["franja_horaria"] = d["hora"].apply(franja_horaria)
    d["es_fin_de_semana"] = np.select(
        [d["dia_semana"].isna(), d["dia_semana"].isin([6, 7])],
        ["Desconocido", "Fin de semana"],
        default="Entre semana",
    )
    d["es_horario_critico"] = (
        (d["franja_horaria"] == "Madrugada") & (d["es_fin_de_semana"] == "Fin de semana")
    ).astype(int)
    d["vehiculo_vulnerable"] = d["tipo_vehiculo"].isin(vehiculos_vulnerables).astype(int)
    d["edad_vulnerable"] = ((d["edad"] < 18) | (d["edad"] >= 60)).astype(int)
    d["vulnerabilidad_vehiculo_edad"] = (d["vehiculo_vulnerable"] & d["edad_vulnerable"]).astype(int)

    feriado, temporada_alta = _feriado_y_temporada_alta(anio, mes, dia)
    d["es_feriado"] = feriado
    d["es_temporada_alta_vial"] = temporada_alta

    # Codificacion ciclica. En el notebook la hora faltante se imputa con la mediana del dataset;
    # aqui, sobre una sola fila, no hay mediana que calcular, asi que se usa el mediodia como
    # valor neutro (el punto mas bajo de la curva de fatalidad por hora, seccion 4.2).
    hora_rellena = d["hora"].fillna(12)
    d["hora_sin"] = np.sin(2 * np.pi * hora_rellena / 24)
    d["hora_cos"] = np.cos(2 * np.pi * hora_rellena / 24)
    d["mes_sin"] = np.sin(2 * np.pi * d["mes"] / 12)
    d["mes_cos"] = np.cos(2 * np.pi * d["mes"] / 12)

    d["municipio_freq"] = d["municipio"].map(art["frecuencias"]["municipio"]).fillna(0)
    d["marca_freq"] = d["marca_vehiculo"].map(art["frecuencias"]["marca_vehiculo"]).fillna(0)

    esperadas = columnas_esperadas(art)
    faltantes = [c for c in esperadas if c not in d.columns]
    if faltantes:
        raise KeyError(
            f"Faltan columnas que el modelo espera: {faltantes}. "
            "Indica que scoring_tfm.py quedo desincronizado de la seccion 5 del notebook."
        )
    return d[esperadas]


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------

def clasificar_nivel(probabilidad):
    """Banda de riesgo y color asociado (Capitulo 6.2)."""
    if probabilidad < UMBRAL_RIESGO_BAJO:
        return "Bajo", "green"
    if probabilidad < UMBRAL_RIESGO_ALTO:
        return "Moderado", "orange"
    return "Alto", "red"


def estimar_riesgo(art, X):
    """Probabilidad estimada de fatalidad y su banda de riesgo para una fila ya construida."""
    X_prep = art["preprocesador"].transform(X)
    probabilidad = float(art["modelo"].predict_proba(X_prep)[0, 1])
    nivel, color = clasificar_nivel(probabilidad)
    return {"probabilidad": probabilidad, "nivel": nivel, "color": color}
