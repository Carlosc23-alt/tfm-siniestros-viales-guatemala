"""
Prototipo de prueba de concepto: Reporte y triage de siniestros viales (sector asegurador)
TFM - Modelado Predictivo de la Severidad y Riesgo de Fatalidad en Siniestros Viales en Guatemala

IMPORTANTE: esto es un PROTOTIPO / prueba de concepto (Capitulo 6.3 de la memoria). No esta
desplegado como servicio, no tiene base de datos multiusuario, y los reportes que guarda NO se
usan para reentrenar el modelo: un reporte tomado en el momento del hecho todavia no sabe si
hubo un fallecido o no, eso solo se confirma despues (ver Capitulo 7.3, verificacion posterior).

Esta version esta enfocada exclusivamente en personal de aseguradora que auxilia el siniestro en
el lugar de los hechos -- alguien capacitado que puede describir con precision el vehiculo y el
tipo de evento.

Diseno del formulario
---------------------
El formulario captura, ademas de los datos descriptivos del siniestro, las variables que el modelo
del TFM necesita para estimar el riesgo (edad y sexo de la persona afectada, departamento,
municipio, tipo de vehiculo, tipo de evento y la hora, que se toma automaticamente). Con eso, el
reporte deja de ser solo captura de datos y se convierte en una herramienta de TRIAGE: al enviarlo,
el prototipo devuelve en el momento la probabilidad estimada de fatalidad, que es lo que permite
priorizar la atencion de un siniestro sobre otro.

Todo lo que no es imprescindible para eso (numero de poliza, comentarios, detalles de la atencion)
esta agrupado en secciones opcionales plegadas, para no alargar la captura en campo.

Como correrlo (dentro de la carpeta del TFM):
    pip install streamlit streamlit-js-eval
    streamlit run reporte_aseguradora.py

Sin `streamlit-js-eval` la app funciona igual, pidiendo la latitud/longitud a mano.
Los artefactos de modelo/ si son obligatorios: de ahi salen tanto el modelo como las
categorias oficiales del INE que pueblan los selectores.
"""

import csv
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Este archivo vive en apps/; el modulo de scoring compartido vive en src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import scoring_tfm

try:
    from streamlit_js_eval import get_geolocation
    GEOLOCALIZACION_DISPONIBLE = True
except ImportError:
    GEOLOCALIZACION_DISPONIBLE = False

st.set_page_config(page_title="Reporte y triage de siniestros - Sector Asegurador",
                   page_icon="📍", layout="centered")

RUTA_CSV = str(Path(__file__).resolve().parent.parent / "reportes_aseguradora.csv")
COLUMNAS = [
    # --- Variables que el modelo actual usa para predecir -------------------
    "timestamp_reporte", "tipo_reportante",
    "latitud", "longitud", "ubicacion_metodo", "departamento", "municipio",
    "tipo_vehiculo", "marca_vehiculo", "tipo_evento", "num_personas_involucradas",
    "edad_afectado", "sexo_afectado",
    # --- Propuesta de captura ampliada (Capitulo 7.4): el modelo actual NO las
    #     usa porque no fue entrenado con ellas; se capturan para demostrar que
    #     son registrables en campo y habilitar modelos futuros ---------------
    "rol_persona", "dispositivo_seguridad", "condicion_iluminacion",
    "estado_superficie", "prueba_alcoholemia", "hora_primer_auxilio",
    # --- Operativas y administrativas --------------------------------------
    "heridos_visibles", "autoridad_en_lugar", "requiere_ambulancia",
    "requiere_grua", "referencia_poliza", "comentario",
    "riesgo_estimado", "nivel_riesgo", "estado",
]

# Opciones de la propuesta de captura ampliada. Cada una responde a un vacio concreto
# identificado en el Capitulo 7.2 y se eligio por ser observable en el lugar del hecho
# sin exigir un juicio tecnico al reportante (ver Capitulo 7.4).
OPCIONES_ROL = ["Conductor", "Pasajero", "Peaton", "Ciclista", "Se desconoce"]
OPCIONES_DISPOSITIVO = ["Cinturon", "Casco", "Ninguno", "No aplica", "Se desconoce"]
OPCIONES_ILUMINACION = ["Luz de dia", "Noche con alumbrado", "Noche sin alumbrado",
                        "Amanecer / atardecer"]
OPCIONES_SUPERFICIE = ["Seca", "Mojada", "En mal estado (baches, grava)", "En obras"]
OPCIONES_ALCOHOLEMIA = ["No se realizo", "Si, realizada por PNC", "Se desconoce"]

TIPO_REPORTANTE_FIJO = "Personal de aseguradora"
LAT_DEFAULT, LON_DEFAULT = 14.6349, -90.5069


# ---------------------------------------------------------------------------
# Artefactos del modelo (opcionales: sin ellos el formulario sigue capturando)
# ---------------------------------------------------------------------------

@st.cache_resource
def cargar_modelo():
    return scoring_tfm.cargar_artefactos()


try:
    ARTEFACTOS = cargar_modelo()
    MODELO_DISPONIBLE = True
except Exception:
    ARTEFACTOS, MODELO_DISPONIBLE = None, False


def sin_ignorado(mapa):
    """Quita la categoria 'Ignorado' del diccionario del INE: no tiene sentido que alguien la
    elija a proposito estando en el lugar de los hechos."""
    return {c: e for c, e in mapa.items() if "ignorado" not in str(e).lower()}


if MODELO_DISPONIBLE:
    D = ARTEFACTOS["diccionarios"]
    MAPA_DEPARTAMENTO = sin_ignorado(D["MAPA_DEPARTAMENTO"])
    MAPA_MUNICIPIO = D["MAPA_MUNICIPIO"]
    MAPA_TIPO_VEHICULO = sin_ignorado(D["MAPA_TIPO_VEHICULO"])
    MAPA_TIPO_EVENTO = sin_ignorado(D["MAPA_TIPO_EVENTO"])
    MAPA_SEXO = sin_ignorado(D["MAPA_SEXO"])
    MUNICIPIOS_POR_DEPARTAMENTO = D.get("MUNICIPIOS_POR_DEPARTAMENTO", {})
    FREQ_MARCA = ARTEFACTOS["frecuencias"]["marca_vehiculo"]
    OPCIONES_MARCA = sorted(
        str(m) for m in FREQ_MARCA.index if pd.notna(m) and "ignorado" not in str(m).lower()
    )


def municipios_del_departamento(cod_depto):
    codigos = MUNICIPIOS_POR_DEPARTAMENTO.get(cod_depto, [])
    pares = sorted(((c, MAPA_MUNICIPIO.get(c, str(c))) for c in codigos), key=lambda x: x[1])
    if pares:
        return pares
    return sorted(MAPA_MUNICIPIO.items(), key=lambda x: x[1])


def cargar_reportes():
    if os.path.exists(RUTA_CSV):
        return pd.read_csv(RUTA_CSV)
    return pd.DataFrame(columns=COLUMNAS)


def guardar_reporte(fila):
    existe = os.path.exists(RUTA_CSV)
    with open(RUTA_CSV, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.title("📍 Reporte y triage de siniestro vial")
st.caption(
    "Prototipo para personal de aseguradora en el lugar de los hechos — TFM, Máster en Big Data, "
    "Data Science e Inteligencia Artificial (UCM)"
)

if MODELO_DISPONIBLE:
    st.info(
        "**Prototipo de prueba de concepto**, no un sistema en producción. Los reportes se guardan "
        f"localmente en `{RUTA_CSV}`. Al enviar, el modelo del TFM estima la probabilidad de "
        "fatalidad del caso para apoyar la priorización de la atención — es una ayuda a la "
        "decisión, no un diagnóstico."
    )
else:
    st.error(
        "No se encontraron los artefactos del modelo en `modelo/`. Este formulario los necesita "
        "no solo para estimar el riesgo, sino también para poblar los selectores con las "
        "categorías oficiales del diccionario del INE. Ejecuta el notebook "
        "`TFM_Analisis_Completo.ipynb` completo (incluida la sección 9) para generarlos."
    )
    st.stop()

# --- 1. Ubicación ----------------------------------------------------------
st.subheader("1. Ubicación")

if "ubicacion" not in st.session_state:
    st.session_state["ubicacion"] = None
    st.session_state["ubicacion_metodo"] = "Manual"

col_a, col_b = st.columns([1, 2])
with col_a:
    detectar = st.button("📍 Detectar ubicación", use_container_width=True,
                         disabled=not GEOLOCALIZACION_DISPONIBLE)
with col_b:
    st.caption("El navegador pedirá permiso para compartir la ubicación."
               if GEOLOCALIZACION_DISPONIBLE else
               "Instala `streamlit-js-eval` para detectarla automáticamente; "
               "mientras tanto, ingrésala abajo.")

if detectar and GEOLOCALIZACION_DISPONIBLE:
    resultado = get_geolocation()
    if resultado and "coords" in resultado:
        st.session_state["ubicacion"] = (resultado["coords"]["latitude"],
                                         resultado["coords"]["longitude"])
        st.session_state["ubicacion_metodo"] = "GPS del navegador"
        st.success(f"Ubicación detectada: {st.session_state['ubicacion'][0]:.5f}, "
                   f"{st.session_state['ubicacion'][1]:.5f}")
    else:
        st.warning("No se pudo obtener la ubicación. Ingrésala manualmente abajo.")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitud", format="%.5f",
                          value=st.session_state["ubicacion"][0] if st.session_state["ubicacion"] else LAT_DEFAULT)
with col2:
    lon = st.number_input("Longitud", format="%.5f",
                          value=st.session_state["ubicacion"][1] if st.session_state["ubicacion"] else LON_DEFAULT)

col3, col4 = st.columns(2)
with col3:
    depto_cod = st.selectbox("Departamento", options=list(MAPA_DEPARTAMENTO.keys()),
                             format_func=lambda c: MAPA_DEPARTAMENTO[c])
with col4:
    municipios = municipios_del_departamento(depto_cod)
    municipio_cod = st.selectbox("Municipio", options=[c for c, _ in municipios],
                                 format_func=lambda c: dict(municipios).get(c, str(c)))

if st.session_state["ubicacion"] or (lat, lon) != (LAT_DEFAULT, LON_DEFAULT):
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=13)

# --- 2. El siniestro -------------------------------------------------------
st.subheader("2. El siniestro")

col5, col6 = st.columns(2)
with col5:
    tipo_evento_cod = st.selectbox("Tipo de evento", options=list(MAPA_TIPO_EVENTO.keys()),
                                   format_func=lambda c: MAPA_TIPO_EVENTO[c])
    tipo_vehiculo_cod = st.selectbox("Tipo de vehículo principal",
                                     options=list(MAPA_TIPO_VEHICULO.keys()),
                                     format_func=lambda c: MAPA_TIPO_VEHICULO[c])
with col6:
    marca = st.selectbox("Marca del vehículo", options=OPCIONES_MARCA,
                         index=OPCIONES_MARCA.index("Toyota") if "Toyota" in OPCIONES_MARCA else 0)
    num_personas = st.number_input("Personas involucradas", min_value=1, max_value=50, value=1)

# --- 3. Persona afectada ---------------------------------------------------
st.subheader("3. Persona afectada principal")
st.caption("La edad y el sexo son, junto con la ubicación y el tipo de evento, las variables que "
           "más pesan en la estimación de riesgo (Capítulo 5 de la memoria).")

col7, col8, col9 = st.columns(3)
with col7:
    edad = st.number_input("Edad (años)", min_value=0, max_value=110, value=30)
with col8:
    sexo_cod = st.selectbox("Sexo", options=list(MAPA_SEXO.keys()),
                            format_func=lambda c: MAPA_SEXO[c])
with col9:
    heridos_visibles = st.selectbox("¿Heridos visibles?", options=["Sí", "No", "No estoy seguro"])

# --- 4. Propuesta de captura ampliada --------------------------------------
st.subheader("4. Captura ampliada (propuesta)")
st.caption(
    "Estos campos **no alimentan la estimación de arriba** — el modelo actual no fue entrenado "
    "con ellos, porque la fuente oficial no los publica. Se incluyen como propuesta concreta de "
    "recolección (Capítulo 7.4 de la memoria): son las variables que el análisis identificó como "
    "ausentes y determinantes, elegidas por ser observables en el lugar sin exigir un juicio "
    "técnico al reportante. Cada una se responde con un toque."
)
st.caption(
    "Todo el formulario, además, es una maqueta de **captura validada**: listas cerradas en vez "
    "de texto libre, ubicación y hora tomadas del dispositivo en vez de transcritas, y la "
    "categoría \"Ignorado\" excluida por diseño. Son las tres cosas que evitan los defectos "
    "documentados en el Capítulo 2.3 de la memoria."
)

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    rol_persona = st.selectbox("Rol de la persona", options=OPCIONES_ROL)
    estado_superficie = st.selectbox("Estado de la superficie", options=OPCIONES_SUPERFICIE)
with col_p2:
    dispositivo = st.selectbox("Dispositivo de seguridad en uso", options=OPCIONES_DISPOSITIVO)
    prueba_alcoholemia = st.selectbox("Prueba de alcoholemia", options=OPCIONES_ALCOHOLEMIA,
                                      help="Se pregunta si la autoridad realizó la prueba, no si "
                                           "hubo consumo: eso es una determinación que no "
                                           "corresponde al reportante.")
with col_p3:
    iluminacion = st.selectbox("Condición de iluminación", options=OPCIONES_ILUMINACION)
    hora_auxilio = st.text_input("Hora de llegada del primer auxilio", placeholder="HH:MM (opcional)",
                                 help="Junto con la hora del reporte permite calcular el tiempo de "
                                      "respuesta, hoy no registrado en la fuente oficial.")

with st.expander("¿Por qué estos campos y no otros?"):
    st.markdown(
        "- **Se incluyen** porque son observables directamente, se responden en segundos y cubren "
        "vacíos que el análisis señaló como determinantes de la severidad: protección del "
        "ocupante, visibilidad, estado de la vía y tiempo de auxilio.\n"
        "- **Se excluye la velocidad estimada**: dos observadores darían cifras distintas y ninguno "
        "puede verificarla. El límite de velocidad de la vía es objetivo y se puede derivar de las "
        "coordenadas, sin preguntarlo.\n"
        "- **Se excluye la causa del siniestro**: es una interpretación, no una observación. Sería "
        "el campo que más se llenaría con \"Ignorado\", que ya es la categoría más problemática "
        "de la base actual.\n"
        "- **Se excluye la gravedad de las lesiones**: es un juicio médico que el reportante no "
        "está capacitado para emitir.\n"
        "- **Se excluyen datos personales identificables** (nombre, documento, placa): el análisis "
        "no los necesita y su captura añadiría obligaciones de protección de datos sin beneficio "
        "analítico.\n\n"
        "Además, **clima, tipo de vía, límite de velocidad y distancia al centro asistencial más "
        "cercano no se preguntan**: son derivables automáticamente de la coordenada y la hora, que "
        "esta app ya captura. Esa es la razón por la que la georreferenciación exacta encabeza las "
        "recomendaciones del Capítulo 7.4.\n\n"
        "Nota sobre el proceso actual: el manual de procesos del INE documenta que las comisarías "
        "trasladan los hechos al Departamento de Estadística de la PNC, que los remite al INE en "
        "hojas de Excel por correo electrónico. El instrumento de captura en el lugar del hecho no "
        "aparece documentado públicamente, así que este formulario no pretende reemplazarlo: "
        "ilustra qué aspecto tendría una captura con validación."
    )

# --- 5. Situación en el lugar (opcional) -----------------------------------
with st.expander("5. Situación en el lugar y datos administrativos (opcional)"):
    col10, col11 = st.columns(2)
    with col10:
        autoridad = st.selectbox("¿Autoridad ya en el lugar?",
                                 options=["No", "PNC", "Bomberos", "PNC y Bomberos"])
        requiere_ambulancia = st.checkbox("Requiere ambulancia")
    with col11:
        referencia_poliza = st.text_input("Referencia de póliza o siniestro", placeholder="Opcional")
        requiere_grua = st.checkbox("Requiere grúa")
    comentario = st.text_area("Comentario adicional",
                              placeholder="Cualquier detalle relevante observado en el lugar...")

st.divider()
enviado = st.button("Enviar reporte y estimar riesgo", use_container_width=True, type="primary")

if enviado:
    ahora = datetime.now()
    X = scoring_tfm.construir_features(
        ARTEFACTOS, edad=edad, sexo=sexo_cod, departamento=depto_cod, municipio=municipio_cod,
        tipo_vehiculo=tipo_vehiculo_cod, tipo_evento=tipo_evento_cod, marca_vehiculo=marca,
        hora=ahora.hour, mes=ahora.month, dia=ahora.day,
        dia_semana=ahora.isoweekday(), anio=ahora.year,
    )
    riesgo = scoring_tfm.estimar_riesgo(ARTEFACTOS, X)

    guardar_reporte({
        "timestamp_reporte": ahora.isoformat(timespec="seconds"),
        "tipo_reportante": TIPO_REPORTANTE_FIJO,
        "latitud": round(lat, 5), "longitud": round(lon, 5),
        "ubicacion_metodo": st.session_state.get("ubicacion_metodo", "Manual"),
        "departamento": MAPA_DEPARTAMENTO.get(depto_cod, depto_cod),
        "municipio": dict(municipios).get(municipio_cod, municipio_cod),
        "tipo_vehiculo": MAPA_TIPO_VEHICULO.get(tipo_vehiculo_cod, tipo_vehiculo_cod),
        "marca_vehiculo": marca,
        "tipo_evento": MAPA_TIPO_EVENTO.get(tipo_evento_cod, tipo_evento_cod),
        "num_personas_involucradas": num_personas,
        "edad_afectado": edad,
        "sexo_afectado": MAPA_SEXO.get(sexo_cod, sexo_cod),
        "rol_persona": rol_persona,
        "dispositivo_seguridad": dispositivo,
        "condicion_iluminacion": iluminacion,
        "estado_superficie": estado_superficie,
        "prueba_alcoholemia": prueba_alcoholemia,
        "hora_primer_auxilio": hora_auxilio.strip(),
        "heridos_visibles": heridos_visibles,
        "autoridad_en_lugar": autoridad,
        "requiere_ambulancia": int(requiere_ambulancia),
        "requiere_grua": int(requiere_grua),
        "referencia_poliza": referencia_poliza.strip(),
        "comentario": comentario.strip(),
        "riesgo_estimado": round(riesgo["probabilidad"], 4),
        "nivel_riesgo": riesgo["nivel"],
        "estado": "pendiente_de_verificacion",
    })

    st.success("Reporte registrado.")
    st.subheader("Triage estimado")
    c1, c2 = st.columns(2)
    c1.metric("Probabilidad estimada de fatalidad", f"{riesgo['probabilidad'] * 100:.1f}%")
    c2.markdown(f"Prioridad sugerida: :{riesgo['color']}[**{riesgo['nivel']}**]")
    st.caption(
        f"Estimación del modelo del TFM ({scoring_tfm.resumen_metricas(ARTEFACTOS)}). Es una ayuda "
        "a la priorización, no un diagnóstico: no incorpora variables que la fuente oficial no "
        "publica (velocidad real, alcohol, clima, estado de la vía) — ver Capítulo 7.2."
    )
    with st.expander("Ver variables enviadas al modelo"):
        st.dataframe(X.T.rename(columns={0: "valor"}))

st.divider()
with st.expander("Últimos reportes registrados (solo para esta demo)"):
    reportes = cargar_reportes()
    if reportes.empty:
        st.caption("Todavía no hay reportes guardados.")
    else:
        st.dataframe(reportes.tail(10).iloc[::-1], use_container_width=True)

st.caption(
    "Proyecto académico — TFM, Máster en Big Data, Data Science e Inteligencia Artificial (UCM). "
    "Prototipo de prueba de concepto, no un sistema en producción."
)
