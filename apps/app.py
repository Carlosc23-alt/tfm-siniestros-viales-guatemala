"""
Simulador de Riesgo de Fatalidad en Siniestros Viales - Guatemala
TFM - Modelado Predictivo de la Severidad y Riesgo de Fatalidad en Siniestros Viales en Guatemala

Como correrlo (en tu máquina, dentro de la carpeta del TFM):
    pip install streamlit
    streamlit run app.py

Requiere que existan (generados por la seccion 9 del notebook TFM_Analisis_Completo.ipynb):
    modelo/preprocesador_tfm.joblib
    modelo/modelo_final_tfm.joblib
    modelo/frecuencias_tfm.joblib
    modelo/diccionarios_tfm.joblib
    modelo/features_tfm.json
"""

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Este archivo vive en apps/; el modulo de scoring compartido vive en src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import scoring_tfm

st.set_page_config(page_title="Simulador de Riesgo Vial - Guatemala", page_icon="🚦", layout="centered")

# ---------------------------------------------------------------------------
# Carga de artefactos (una sola vez, cacheada)
# ---------------------------------------------------------------------------


@st.cache_resource
def cargar_artefactos():
    """Delegado a scoring_tfm, el modulo compartido con reporte_aseguradora.py, para que la
    ingenieria de caracteristicas y la lista de variables del modelo no puedan desincronizarse
    entre las dos apps ni respecto al notebook."""
    return scoring_tfm.cargar_artefactos()


try:
    ARTEFACTOS = cargar_artefactos()
except FileNotFoundError as e:
    st.error(
        "No se encontraron los artefactos del modelo. Corre primero el notebook "
        "`TFM_Analisis_Completo.ipynb` completo (incluida la seccion 9) para generar la carpeta "
        f"`modelo/`.\n\nDetalle: {e}"
    )
    st.stop()

preprocessor = ARTEFACTOS["preprocesador"]
modelo = ARTEFACTOS["modelo"]
frecuencias = ARTEFACTOS["frecuencias"]
diccionarios = ARTEFACTOS["diccionarios"]
features = ARTEFACTOS["features"]

MAPA_DEPARTAMENTO = diccionarios["MAPA_DEPARTAMENTO"]
MAPA_MUNICIPIO = diccionarios["MAPA_MUNICIPIO"]
MAPA_TIPO_VEHICULO = diccionarios["MAPA_TIPO_VEHICULO"]
MAPA_TIPO_EVENTO = diccionarios["MAPA_TIPO_EVENTO"]
MAPA_SEXO = diccionarios["MAPA_SEXO"]
MAPA_DIA_SEMANA = diccionarios["MAPA_DIA_SEMANA"]
VEHICULOS_VULNERABLES = diccionarios["VEHICULOS_VULNERABLES"]
# Requiere haber vuelto a correr la Seccion 9 del notebook despues del 30/08 (agrega el
# departamento -> municipios que la app usa para filtrar el selectbox de municipio).
MUNICIPIOS_POR_DEPARTAMENTO = diccionarios.get("MUNICIPIOS_POR_DEPARTAMENTO", {})

FREQ_MUNICIPIO = frecuencias["municipio"]
FREQ_MARCA = frecuencias["marca_vehiculo"]
NUM_FEATURES = features["num_features"]
CAT_FEATURES = features["cat_features"]

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

# Municipios/marcas a mostrar en los selectbox: solo las que el modelo realmente vio en
# entrenamiento (estan en el indice de la serie de frecuencias), ordenadas por nombre.
TODOS_LOS_MUNICIPIOS = sorted(
    [(cod, MAPA_MUNICIPIO.get(cod, str(cod))) for cod in FREQ_MUNICIPIO.index],
    key=lambda x: x[1],
)
OPCIONES_MARCA = sorted(
    [str(m) for m in FREQ_MARCA.index if pd.notna(m) and "ignorado" not in str(m).lower()]
)


def municipios_del_departamento(departamento_cod):
    """Codigos de municipio conocidos por el modelo que pertenecen a ese departamento,
    ordenados por nombre. Si el departamento no tiene municipios registrados en el
    diccionario (por ejemplo, un diccionario generado antes de este cambio), muestra
    todos los municipios en vez de dejar el selectbox vacio."""
    codigos_validos = set(MUNICIPIOS_POR_DEPARTAMENTO.get(departamento_cod, []))
    opciones = [par for par in TODOS_LOS_MUNICIPIOS if par[0] in codigos_validos]
    return opciones if opciones else TODOS_LOS_MUNICIPIOS


def sin_ignorado(mapa):
    """Quita del selectbox la categoria 'Ignorado' del diccionario del INE. El modelo si vio esta
    categoria en el entrenamiento (es un valor real de los datos) y la sigue manejando bien si
    algun dia se le pasa, pero no tiene sentido dejar que alguien arme a proposito un escenario
    hipotetico con sexo/tipo de vehiculo/tipo de evento "desconocido" -- solo ensucia el desplegable."""
    return {codigo: etiqueta for codigo, etiqueta in mapa.items() if "ignorado" not in etiqueta.lower()}

# ---------------------------------------------------------------------------
# Funciones de feature engineering -- deben ser un espejo exacto de
# `aplicar_feature_engineering()` en el notebook (seccion 5), para que la app y el
# entrenamiento nunca diverjan. Si cambias esa funcion en el notebook, cambia esta tambien.
# ---------------------------------------------------------------------------


def franja_horaria(hora):
    if pd.isna(hora):
        return "Desconocido"
    hora = int(hora)
    if 0 <= hora < 6:
        return "Madrugada"
    if 6 <= hora < 12:
        return "Manana"
    if 12 <= hora < 18:
        return "Tarde"
    return "Noche"


def construir_features(edad, sexo_cod, departamento_cod, municipio_cod, tipo_vehiculo_cod,
                        tipo_evento_cod, marca, hora, mes, dia, dia_semana_cod):
    """Delegado a scoring_tfm.construir_features (misma logica que la seccion 5 del notebook,
    incluidas es_feriado y es_temporada_alta_vial)."""
    return scoring_tfm.construir_features(
        ARTEFACTOS, edad=edad, sexo=sexo_cod, departamento=departamento_cod,
        municipio=municipio_cod, tipo_vehiculo=tipo_vehiculo_cod, tipo_evento=tipo_evento_cod,
        marca_vehiculo=marca, hora=hora, mes=mes, dia=dia, dia_semana=dia_semana_cod,
    )


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.title("🚦 Simulador de Riesgo de Fatalidad en Siniestros Viales")
st.caption(
    "TFM — Modelado Predictivo de la Severidad y Riesgo de Fatalidad en Siniestros Viales en "
    "Guatemala · Universidad Complutense de Madrid"
)

st.markdown(
    "Completa las variables de un siniestro para estimar la probabilidad de que derive en un "
    "resultado fatal, según el modelo `LightGBM` entrenado sobre datos oficiales del INE/PNC "
    "(2018-2023) y evaluado sobre 2024."
)

# Nota tecnica: el formulario ya NO usa st.form. Un st.form agrupa los campos y solo reacciona
# al enviarse, lo cual es mas prolijo mientras los campos son independientes -- pero aqui
# necesitamos que el selectbox de municipio se actualice EN VIVO en cuanto cambia el
# departamento (para no dejar seleccionar un municipio que no le pertenece), y eso solo pasa
# con widgets fuera de un formulario. El boton de abajo cumple el mismo rol que antes.
col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad de la persona involucrada", min_value=0, max_value=100, value=30)
    sexo_cod = st.selectbox(
        "Sexo", options=list(sin_ignorado(MAPA_SEXO).keys()), format_func=lambda c: MAPA_SEXO[c]
    )
    departamento_cod = st.selectbox(
        "Departamento", options=list(MAPA_DEPARTAMENTO.keys()), format_func=lambda c: MAPA_DEPARTAMENTO[c]
    )
    opciones_municipio = municipios_del_departamento(departamento_cod)
    if not MUNICIPIOS_POR_DEPARTAMENTO:
        st.caption(
            "⚠️ Estos artefactos no traen todavía el filtro de municipios por departamento "
            "(vuelve a correr la Sección 9 del notebook y reinicia la app) — por ahora se listan "
            "todos los municipios."
        )
    municipio_cod = st.selectbox(
        "Municipio", options=[c for c, _ in opciones_municipio], format_func=lambda c: MAPA_MUNICIPIO.get(c, str(c))
    )
    tipo_vehiculo_cod = st.selectbox(
        "Tipo de vehículo",
        options=list(sin_ignorado(MAPA_TIPO_VEHICULO).keys()),
        format_func=lambda c: MAPA_TIPO_VEHICULO[c],
    )
    marca = st.selectbox("Marca del vehículo", options=OPCIONES_MARCA)

with col2:
    tipo_evento_cod = st.selectbox(
        "Tipo de evento",
        options=list(sin_ignorado(MAPA_TIPO_EVENTO).keys()),
        format_func=lambda c: MAPA_TIPO_EVENTO[c],
    )
    hora = st.slider("Hora del día (0-23)", min_value=0, max_value=23, value=18)
    mes = st.selectbox("Mes", options=list(range(1, 13)), format_func=lambda m: MESES_ES[m])
    dia = st.number_input("Día del mes", min_value=1, max_value=31, value=15)
    dia_semana_cod = st.selectbox(
        "Día de la semana", options=list(MAPA_DIA_SEMANA.keys()), format_func=lambda c: MAPA_DIA_SEMANA[c]
    )

enviado = st.button("Estimar riesgo", use_container_width=True, type="primary")

if enviado:
    X = construir_features(
        edad, sexo_cod, departamento_cod, municipio_cod, tipo_vehiculo_cod,
        tipo_evento_cod, marca, hora, mes, dia, dia_semana_cod,
    )
    X_prep = preprocessor.transform(X)
    probabilidad = modelo.predict_proba(X_prep)[0, 1]

    if probabilidad < 0.20:
        nivel, color = "Bajo", "green"
    elif probabilidad < 0.50:
        nivel, color = "Moderado", "orange"
    else:
        nivel, color = "Alto", "red"

    st.divider()
    st.subheader("Resultado de la simulación")
    c1, c2 = st.columns(2)
    c1.metric("Probabilidad estimada de fatalidad", f"{probabilidad * 100:.1f}%")
    c2.markdown(f"Nivel de riesgo: :{color}[**{nivel}**]")

    st.caption(
        "Esta estimación proviene de un modelo entrenado sobre datos históricos administrativos "
        "del INE/PNC (2018-2023) y validado fuera de tiempo sobre 2024 "
        f"({scoring_tfm.resumen_metricas(ARTEFACTOS)}). "
        "No sustituye una evaluación profesional de riesgo ni incorpora variables no "
        "disponibles en la fuente (velocidad real, alcohol, clima, estado de la vía) — ver "
        "limitaciones en el Capítulo 7 de la memoria."
    )

    with st.expander("Ver variables enviadas al modelo"):
        st.dataframe(X.T.rename(columns={0: "valor"}))

st.divider()
st.caption(
    "Proyecto académico — TFM, Máster en Big Data, Data Science e Inteligencia Artificial (UCM). "
    "No constituye asesoría de seguridad vial ni de suscripción de seguros."
)
