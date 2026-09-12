"""
Prototipo de prueba de concepto: Reporte y triage de siniestros viales (sector asegurador)
TFM - Modelado Predictivo de la Severidad y Riesgo de Fatalidad en Siniestros Viales en Guatemala

IMPORTANTE: esto es un PROTOTIPO / prueba de concepto (Capitulo 6.3 de la memoria). Los
reportes que guarda NO se usan para reentrenar el modelo: un reporte tomado en el momento del
hecho todavia no sabe si hubo un fallecido o no -- eso solo se confirma despues, de ahi que
cada fila quede marcada como `pendiente_de_verificacion` (ver Capitulo 7.4).

Esta enfocado en personal de aseguradora que auxilia el siniestro en el lugar de los hechos:
alguien capacitado que puede describir con precision el vehiculo y el tipo de evento.

El grano del registro: una persona, no un siniestro
---------------------------------------------------
La fuente del INE tiene una fila por PERSONA fallecida o lesionada, y el modelo estima
exactamente eso: la probabilidad de que una persona involucrada resulte fallecida. El
formulario replica ese grano -- el sujeto de cada reporte es el CONDUCTOR del vehiculo
reportado, de modo que una colision entre dos vehiculos produce dos reportes. Los demas
afectados (acompanantes, peatones, ocupantes del otro vehiculo) se cuentan en el bloque
"otros afectados", lo que permite reconstruir el total del siniestro sin crear registros
incompletos. Como consecuencia, el "rol de la persona" no se pregunta: queda codificado en
la estructura del registro, que es mas fiable que un campo que podria quedar en "Se
desconoce".

Las dos funciones del prototipo
-------------------------------
1. TRIAGE: al enviar, devuelve la probabilidad estimada de fatalidad, que permite priorizar
   la atencion de un siniestro sobre otro.
2. MAQUETA DE CAPTURA VALIDADA: listas cerradas en vez de texto libre, ubicacion y hora
   tomadas del dispositivo en vez de transcritas, y la categoria "Ignorado" excluida por
   diseno -- las tres cosas que evitan los defectos documentados en el Capitulo 2.3. Los
   campos marcados con una daga son la propuesta de captura del Capitulo 7.4: el modelo
   actual no los usa y la interfaz lo advierte.

Como correrlo
-------------
    pip install -r requirements.txt
    streamlit run apps/reporte_aseguradora.py     # desde la raiz del repositorio

Funciona igual si el archivo esta en una carpeta plana junto a scoring_tfm.py.

Dependencias opcionales: sin `streamlit-js-eval` no hay boton de geolocalizacion y la
latitud/longitud se escriben a mano; sin `requests` no se deducen departamento y municipio
desde la coordenada. Los artefactos de modelo/ si son obligatorios: de ahi salen tanto el
modelo como las categorias oficiales del INE que pueblan los selectores.
"""

import csv
import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Raiz del proyecto. Este archivo funciona igual en el repositorio (donde vive en apps/ y
# el modulo de scoring en src/) que en una carpeta plana con todo junto: src/ se anade al
# path solo si existe, asi que mover el archivo de un sitio al otro no rompe el import.
RAIZ = Path(__file__).resolve().parent.parent
if (RAIZ / "src").is_dir():
    sys.path.insert(0, str(RAIZ / "src"))

import scoring_tfm

try:
    from streamlit_js_eval import get_geolocation
    GEOLOCALIZACION_DISPONIBLE = True
except ImportError:
    GEOLOCALIZACION_DISPONIBLE = False

st.set_page_config(page_title="Reporte y triage de siniestros - Sector Asegurador",
                   page_icon="📍", layout="centered")

# Se ancla a la raiz del proyecto, no al directorio desde el que se invoca streamlit.
RUTA_CSV = str(RAIZ / "reportes_aseguradora.csv")
# Un registro = UNA persona, igual que en la fuente del INE, donde cada fila es una
# persona fallecida o lesionada. El sujeto del registro es el CONDUCTOR del vehiculo
# reportado; en una colision entre dos vehiculos se llenan dos reportes. Los demas
# afectados (acompanantes, peatones) se cuentan en el bloque "otros afectados", que es
# lo que permite reconstruir el total del siniestro sin inventar registros incompletos.
COLUMNAS = [
    "timestamp_reporte", "tipo_reportante",
    # --- Donde y cuando -----------------------------------------------------
    "latitud", "longitud", "ubicacion_metodo", "departamento", "municipio",
    # --- El siniestro -------------------------------------------------------
    "tipo_vehiculo", "marca_vehiculo", "tipo_evento",
    # --- El conductor: sujeto del registro ----------------------------------
    "edad_conductor", "sexo_conductor", "desenlace_conductor",
    # --- Propuesta de captura (Capitulo 7.4): el modelo actual NO usa estas
    #     variables porque no fue entrenado con ellas; se capturan para mostrar
    #     que son registrables en campo y habilitar modelos futuros ----------
    "dispositivo_seguridad", "condicion_iluminacion", "estado_superficie",
    "prueba_alcoholemia", "hora_primer_auxilio",
    # --- Otros afectados del mismo siniestro --------------------------------
    "hubo_otros_afectados", "otros_afectados_num", "otros_afectados_fallecidos",
    "otros_afectados_lesionados", "otros_afectados_tipo",
    "num_personas_involucradas",
    # --- Operativas y administrativas --------------------------------------
    "autoridad_en_lugar", "requiere_ambulancia", "requiere_grua",
    "referencia_poliza", "comentario",
    # --- Salida del modelo y estado del registro ----------------------------
    "riesgo_estimado", "nivel_riesgo", "estado",
]

# Opciones de la propuesta de captura ampliada. Cada una responde a un vacio concreto
# identificado en el Capitulo 7.2 y se eligio por ser observable en el lugar del hecho
# sin exigir un juicio tecnico al reportante (ver Capitulo 7.4).
# El estado se OBSERVA en el lugar; no es el desenlace final, que puede cambiar despues
# (de ahi que cada reporte quede como pendiente_de_verificacion).
OPCIONES_DESENLACE = ["Sin lesiones aparentes", "Lesionado", "Fallecido en el lugar",
                      "Se desconoce"]
OPCIONES_OTROS_TIPO = ["Acompanantes del mismo vehiculo", "Peatones",
                       "Ocupantes de otro vehiculo", "Mezcla de varios"]
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


@st.cache_data(ttl=3600, show_spinner=False)
def _geocodificacion_inversa(lat, lon):
    """Nombres administrativos de una coordenada segun OpenStreetMap, o (None, None).

    Se usa Nominatim (la API publica de OSM). Es una dependencia de red y puede fallar o
    estar limitada por cuota: toda la funcion esta pensada para degradar en silencio y
    dejar los selectores como estaban, nunca para bloquear la captura.
    """
    try:
        import requests
        respuesta = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10,
                    "accept-language": "es"},
            headers={"User-Agent": "TFM-siniestros-viales-gt/1.0 (proyecto academico UCM)"},
            timeout=8,
        )
        respuesta.raise_for_status()
        direccion = respuesta.json().get("address", {})
    except Exception:
        return None, None
    departamento = direccion.get("state") or direccion.get("province") or direccion.get("region")
    municipio = (direccion.get("city") or direccion.get("town")
                 or direccion.get("municipality") or direccion.get("village")
                 or direccion.get("county"))
    return departamento, municipio


def _normalizar(texto):
    """Minusculas sin acentos y sin el prefijo 'departamento de', para poder comparar."""
    limpio = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    limpio = limpio.lower().replace("departamento de", "").replace("municipio de", "")
    return " ".join(limpio.split())


def codigos_desde_coordenada(lat, lon):
    """Codigos INE de departamento y municipio para una coordenada.

    Devuelve (depto_cod, municipio_cod); cualquiera puede ser None si el nombre que
    devuelve OSM no coincide con el catalogo oficial del INE. Se compara contra los
    diccionarios del propio modelo, de modo que no se introduce ninguna tabla de
    nombres paralela que pudiera desincronizarse.
    """
    depto_txt, muni_txt = _geocodificacion_inversa(lat, lon)
    if not depto_txt:
        return None, None
    depto_cod = next((c for c, nombre in MAPA_DEPARTAMENTO.items()
                      if _normalizar(nombre) == _normalizar(depto_txt)), None)
    if depto_cod is None:
        return None, None
    muni_cod = None
    if muni_txt:
        muni_cod = next((c for c, nombre in municipios_del_departamento(depto_cod)
                         if _normalizar(nombre) == _normalizar(muni_txt)), None)
    return depto_cod, muni_cod


def cargar_reportes():
    if os.path.exists(RUTA_CSV):
        return pd.read_csv(RUTA_CSV)
    return pd.DataFrame(columns=COLUMNAS)


def _cabecera_existente():
    """Cabecera del CSV actual, o None si no hay archivo."""
    if not os.path.exists(RUTA_CSV):
        return None
    with open(RUTA_CSV, newline="", encoding="utf-8") as f:
        return next(csv.reader(f), None)


def guardar_reporte(fila):
    cabecera = _cabecera_existente()
    # Si el esquema cambio, se aparta el archivo anterior en vez de escribir filas
    # desalineadas encima de el.
    if cabecera is not None and cabecera != COLUMNAS:
        sello = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.rename(RUTA_CSV, RUTA_CSV[:-4] + f"_esquema_anterior_{sello}.csv")
        cabecera = None
    with open(RUTA_CSV, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        if cabecera is None:
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

# En Streamlit Community Cloud el sistema de archivos del contenedor es efimero: se
# reinicia cuando la app duerme por inactividad o cuando se despliega una version nueva.
# Detectarlo permite decir la verdad sobre la persistencia en vez de prometer lo que no hay.
EN_LA_NUBE = os.environ.get("HOSTNAME", "").startswith("streamlit") or os.path.isdir("/mount/src")

if MODELO_DISPONIBLE:
    st.info(
        "**Prototipo de prueba de concepto**, no un sistema en producción. Al enviar, el modelo "
        "del TFM estima la probabilidad de fatalidad del caso para apoyar la priorización de la "
        "atención — es una ayuda a la decisión, no un diagnóstico."
    )
    if EN_LA_NUBE:
        st.warning(
            "**Los reportes de esta demo no se conservan.** La aplicación corre en un contenedor "
            "efímero: cuando se reinicia por inactividad o por un despliegue nuevo, el archivo de "
            "reportes desaparece. Es deliberado — el prototipo existe para mostrar *qué* debería "
            "capturarse y *cómo* validarlo (Capítulo 7.4), no para almacenar datos reales. Una "
            "versión en producción escribiría en una base de datos y añadiría el paso de "
            "verificación posterior del desenlace."
        )
    else:
        st.caption(f"Los reportes se guardan en `{RUTA_CSV}`.")
else:
    st.error(
        "No se encontraron los artefactos del modelo en `modelo/`. Este formulario los necesita "
        "no solo para estimar el riesgo, sino también para poblar los selectores con las "
        "categorías oficiales del diccionario del INE. Ejecuta el notebook "
        "`TFM_Analisis_Completo.ipynb` completo (incluida la sección 9) para generarlos."
    )
    st.stop()

# --- 1. Donde y cuando -----------------------------------------------------
st.subheader("1. Dónde y cuándo")
st.caption("La fecha y la hora se toman automáticamente del dispositivo al enviar el reporte; "
           "no hay que escribirlas.")

if "ubicacion" not in st.session_state:
    st.session_state["ubicacion"] = None
    st.session_state["ubicacion_metodo"] = "Manual"
    st.session_state["esperando_gps"] = False
# Los number_input de lat/lon se gobiernan por clave, para poder rellenarlos desde el GPS:
# una vez que un widget existe, Streamlit ignora su argumento `value=` en los reruns.
st.session_state.setdefault("lat_widget", LAT_DEFAULT)
st.session_state.setdefault("lon_widget", LON_DEFAULT)

def _autocompletar_division(lat, lon):
    """Rellena los selectores de departamento y municipio desde la coordenada."""
    depto_cod, muni_cod = codigos_desde_coordenada(lat, lon)
    if depto_cod is None:
        st.session_state["division_auto"] = "sin_match"
        return
    st.session_state["depto_sel"] = depto_cod
    if muni_cod is not None:
        st.session_state["muni_sel"] = muni_cod
    st.session_state["division_auto"] = ("completa" if muni_cod is not None
                                         else "solo_departamento")


col_a, col_b = st.columns([1, 2])
with col_a:
    detectar = st.button("📍 Detectar ubicación", use_container_width=True,
                         disabled=not GEOLOCALIZACION_DISPONIBLE)
with col_b:
    st.caption("El navegador pedirá permiso para compartir la ubicación."
               if GEOLOCALIZACION_DISPONIBLE else
               "Instala `streamlit-js-eval` para detectarla automáticamente; "
               "mientras tanto, ingrésala abajo.")

if detectar:
    st.session_state["esperando_gps"] = True

# get_geolocation() es un componente de navegador: en la ejecucion en que se crea devuelve
# None y el valor real llega en un rerun POSTERIOR. Por eso hay que seguir llamandolo
# mientras se espera la respuesta, no solo en la ejecucion en que se pulso el boton.
if st.session_state["esperando_gps"] and GEOLOCALIZACION_DISPONIBLE:
    resultado = get_geolocation()
    if resultado and isinstance(resultado, dict) and resultado.get("coords"):
        st.session_state["lat_widget"] = float(resultado["coords"]["latitude"])
        st.session_state["lon_widget"] = float(resultado["coords"]["longitude"])
        st.session_state["ubicacion"] = (st.session_state["lat_widget"],
                                         st.session_state["lon_widget"])
        st.session_state["ubicacion_metodo"] = "GPS del navegador"
        st.session_state["esperando_gps"] = False
        _autocompletar_division(st.session_state["lat_widget"], st.session_state["lon_widget"])
        st.rerun()
    else:
        st.info("Esperando la respuesta del navegador… si no aparece, revisa que hayas "
                "concedido el permiso de ubicación y que la página se sirva por `localhost` "
                "o por HTTPS (los navegadores bloquean la geolocalización en HTTP).")

if st.session_state["ubicacion_metodo"] == "GPS del navegador":
    st.success(f"Ubicación detectada por GPS: {st.session_state['lat_widget']:.5f}, "
               f"{st.session_state['lon_widget']:.5f}")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitud", format="%.5f", key="lat_widget")
with col2:
    lon = st.number_input("Longitud", format="%.5f", key="lon_widget")

col_g1, col_g2 = st.columns([1, 2])
with col_g1:
    rellenar = st.button("\U0001F5FA\uFE0F Deducir departamento y municipio",
                         use_container_width=True)
with col_g2:
    st.caption("Los deduce de la coordenada consultando OpenStreetMap. Si no hay coincidencia "
               "exacta con el catálogo del INE, los selectores se quedan como están.")
if rellenar:
    _autocompletar_division(lat, lon)

_aviso = st.session_state.pop("division_auto", None)
if _aviso == "completa":
    st.success("Departamento y municipio deducidos de la coordenada. Verificalos antes de enviar.")
elif _aviso == "solo_departamento":
    st.warning("Se dedujo el departamento, pero el municipio que devolvió OpenStreetMap no coincide "
               "con ningún nombre del catálogo del INE. Elegilo a mano.")
elif _aviso == "sin_match":
    st.warning("No se pudo deducir la división administrativa de esa coordenada (sin respuesta del "
               "servicio, o un nombre que no coincide con el catálogo del INE). Elegila a mano.")

col3, col4 = st.columns(2)
with col3:
    depto_cod = st.selectbox("Departamento", options=list(MAPA_DEPARTAMENTO.keys()),
                             format_func=lambda c: MAPA_DEPARTAMENTO[c], key="depto_sel")
with col4:
    municipios = municipios_del_departamento(depto_cod)
    opciones_muni = [c for c, _ in municipios]
    # Si el departamento cambio, el municipio preseleccionado puede haber quedado fuera de
    # la lista: hay que soltarlo antes de crear el widget o Streamlit lanza excepcion.
    if st.session_state.get("muni_sel") not in opciones_muni:
        st.session_state.pop("muni_sel", None)
    municipio_cod = st.selectbox("Municipio", options=opciones_muni,
                                 format_func=lambda c: dict(municipios).get(c, str(c)),
                                 key="muni_sel")

if st.session_state["ubicacion"] or (lat, lon) != (LAT_DEFAULT, LON_DEFAULT):
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=13)

# --- Leyenda del marcador de campos propuestos -----------------------------
MARCA = "\u2020"   # daga: senala los campos que el modelo actual NO usa (Capitulo 7.4)

st.caption(
    f"Los campos marcados con **{MARCA}** son una **propuesta de captura**: el modelo actual no "
    "fue entrenado con ellos, porque la fuente oficial no los publica, de modo que **no alteran "
    "la estimación de riesgo**. Se piden porque el análisis los identificó como determinantes "
    "ausentes y son observables en el lugar con un toque (Capítulo 7.4 de la memoria)."
)

# --- 2. Qué pasó -----------------------------------------------------------
st.subheader("2. Qué pasó")

col5, col6 = st.columns(2)
with col5:
    tipo_evento_cod = st.selectbox("Tipo de evento", options=list(MAPA_TIPO_EVENTO.keys()),
                                   format_func=lambda c: MAPA_TIPO_EVENTO[c])
    tipo_vehiculo_cod = st.selectbox("Tipo de vehículo del conductor reportado",
                                     options=list(MAPA_TIPO_VEHICULO.keys()),
                                     format_func=lambda c: MAPA_TIPO_VEHICULO[c])
with col6:
    marca = st.selectbox("Marca del vehículo", options=OPCIONES_MARCA,
                         index=OPCIONES_MARCA.index("Toyota") if "Toyota" in OPCIONES_MARCA else 0)
    iluminacion = st.selectbox(f"Condición de iluminación {MARCA}", options=OPCIONES_ILUMINACION)
    estado_superficie = st.selectbox(f"Estado de la superficie {MARCA}", options=OPCIONES_SUPERFICIE)

# --- 3. El conductor -------------------------------------------------------
st.subheader("3. El conductor")
st.caption("**Un reporte = un conductor.** En una colisión entre dos vehículos se llenan dos "
           "reportes, uno por cada conductor. Los acompañantes y los peatones se cuentan en la "
           "sección siguiente, sin abrir un reporte propio.")

col7, col8 = st.columns(2)
with col7:
    edad = st.number_input("Edad del conductor (años)", min_value=0, max_value=110, value=30)
    desenlace = st.selectbox("Estado al momento del reporte", options=OPCIONES_DESENLACE,
                             help="Es lo que se OBSERVA en el lugar, no el desenlace final: "
                                  "una persona lesionada puede fallecer despues. Por eso el "
                                  "reporte queda como pendiente de verificacion.")
with col8:
    sexo_cod = st.selectbox("Sexo", options=list(MAPA_SEXO.keys()),
                            format_func=lambda c: MAPA_SEXO[c])
    dispositivo = st.selectbox(f"Dispositivo de seguridad en uso {MARCA}",
                               options=OPCIONES_DISPOSITIVO)

# --- 4. Otros afectados ----------------------------------------------------
st.subheader("4. Otros afectados")

hubo_otros = st.checkbox("Hubo otras personas afectadas en este siniestro")

if hubo_otros:
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        otros_num = st.number_input("¿Cuántas, además del conductor?",
                                    min_value=1, max_value=60, value=1)
        otros_tipo = st.selectbox("¿Quiénes eran?", options=OPCIONES_OTROS_TIPO)
    with col_o2:
        otros_fallecidos = st.number_input("De esas, fallecidas en el lugar",
                                           min_value=0, max_value=60, value=0)
        otros_lesionados = st.number_input("De esas, lesionadas",
                                           min_value=0, max_value=60, value=1)
    if otros_fallecidos + otros_lesionados > otros_num:
        st.warning(f"Fallecidas más lesionadas ({otros_fallecidos + otros_lesionados}) supera el "
                   f"total de otras personas afectadas ({otros_num}). Revisá las cifras antes de "
                   "enviar.")
else:
    otros_num = 0
    otros_tipo = ""
    otros_fallecidos = 0
    otros_lesionados = 0
    st.caption("Si el conductor fue el único afectado, dejá esto sin marcar.")

num_personas = 1 + int(otros_num)

# --- 5. Atención y autoridad -----------------------------------------------
st.subheader("5. Atención y autoridad")

col9, col10 = st.columns(2)
with col9:
    autoridad = st.selectbox("¿Autoridad ya en el lugar?",
                             options=["No", "PNC", "Bomberos", "PNC y Bomberos"])
    prueba_alcoholemia = st.selectbox(f"Prueba de alcoholemia al conductor {MARCA}",
                                      options=OPCIONES_ALCOHOLEMIA,
                                      help="Se pregunta si la autoridad realizó la prueba, no si "
                                           "hubo consumo: eso es una determinación que no "
                                           "corresponde al reportante.")
with col10:
    hora_auxilio = st.text_input(f"Hora de llegada del primer auxilio {MARCA}",
                                 placeholder="HH:MM (opcional)",
                                 help="Junto con la hora del reporte permite calcular el tiempo de "
                                      "respuesta, hoy no registrado en la fuente oficial.")
    requiere_ambulancia = st.checkbox("Requiere ambulancia")
    requiere_grua = st.checkbox("Requiere grúa")

# --- 6. Datos administrativos (opcional) -----------------------------------
with st.expander("6. Datos administrativos (opcional)"):
    referencia_poliza = st.text_input("Referencia de póliza o siniestro", placeholder="Opcional")
    comentario = st.text_area("Comentario adicional",
                              placeholder="Cualquier detalle relevante observado en el lugar...")

# --- Justificacion del diseno del formulario -------------------------------
with st.expander(f"Sobre el diseño de este formulario (y los campos {MARCA})"):
    st.markdown(
        f"Todo el formulario es, además de una herramienta de captura, una maqueta de **captura "
        "validada**: listas cerradas en vez de texto libre, ubicación y hora tomadas del "
        "dispositivo en vez de transcritas, y la categoría \"Ignorado\" excluida por diseño. Son "
        "las tres cosas que evitan los defectos documentados en el Capítulo 2.3 de la memoria "
        "(`hora = 99`, `edad = 999`, un registro con `anio = 3`, el 31.2% sin día de la semana).\n\n"
        "**Por qué un reporte por conductor.** La fuente del INE tiene una fila por *persona* "
        "fallecida o lesionada, no por siniestro, y el modelo estima precisamente la probabilidad "
        "de que **una persona** involucrada resulte fallecida. El formulario replica ese grano: el "
        "sujeto del registro es el conductor, y en una colisión se llenan dos reportes. Eso hace "
        "que el **rol de la persona** —una de las seis variables propuestas en el Capítulo 7.4— no "
        "necesite preguntarse: queda codificado en la estructura misma del registro, que es más "
        "fiable que un campo que el digitador podría dejar en \"Se desconoce\".\n\n"
        f"**Por qué estos campos {MARCA} y no otros:**\n"
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

st.divider()
enviado = st.button("Enviar reporte", use_container_width=True, type="primary")

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
        "edad_conductor": edad,
        "sexo_conductor": MAPA_SEXO.get(sexo_cod, sexo_cod),
        "desenlace_conductor": desenlace,
        "dispositivo_seguridad": dispositivo,
        "condicion_iluminacion": iluminacion,
        "estado_superficie": estado_superficie,
        "prueba_alcoholemia": prueba_alcoholemia,
        "hora_primer_auxilio": hora_auxilio.strip(),
        "hubo_otros_afectados": int(hubo_otros),
        "otros_afectados_num": int(otros_num),
        "otros_afectados_fallecidos": int(otros_fallecidos),
        "otros_afectados_lesionados": int(otros_lesionados),
        "otros_afectados_tipo": otros_tipo,
        "num_personas_involucradas": num_personas,
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
    if desenlace == "Fallecido en el lugar":
        st.info(
            "El conductor consta como fallecido en el lugar, de modo que para este caso el score "
            "ya no sirve para priorizar la atención. Sigue siendo útil para otra cosa: comparar lo "
            "que el modelo habría estimado contra lo que efectivamente ocurrió, que es la vía de "
            "verificación del Capítulo 7.4."
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
