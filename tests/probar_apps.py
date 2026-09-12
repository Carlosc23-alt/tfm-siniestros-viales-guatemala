"""
probar_apps.py -- Prueba de humo de las dos apps de Streamlit y de la capa de scoring.

Verifica, sin intervencion manual:
  1. Que los artefactos de modelo/ cargan.
  2. Que scoring_tfm construye exactamente las columnas que el modelo espera.
  3. Que una prediccion real corre de punta a punta (incluidos los dos escenarios
     que la memoria cita en el Capitulo 6.2).
  4. Que app.py levanta sin excepciones y produce una prediccion al pulsar el boton.
  5. Que reporte_aseguradora.py levanta sin excepciones.

Uso:
    python probar_apps.py

Escribe el resultado en reporte_pruebas.txt y lo imprime en pantalla.
"""

import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Este archivo vive en tests/; el modulo de scoring vive en src/ y las apps en apps/.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

LINEAS = []
FALLOS = 0


def registrar(texto=""):
    print(texto)
    LINEAS.append(texto)


def seccion(titulo):
    registrar()
    registrar("=" * 72)
    registrar(titulo)
    registrar("=" * 72)


def ok(msg):
    registrar(f"  [OK]    {msg}")


def fallo(msg, exc=None):
    global FALLOS
    FALLOS += 1
    registrar(f"  [FALLA] {msg}")
    if exc is not None:
        for linea in traceback.format_exception_only(type(exc), exc):
            registrar(f"          {linea.rstrip()}")


registrar("PRUEBA DE HUMO -- apps del TFM")

# ---------------------------------------------------------------- entorno
seccion("1. Entorno")
try:
    import joblib, numpy, pandas, sklearn
    registrar(f"  python       {sys.version.split()[0]}")
    registrar(f"  scikit-learn {sklearn.__version__}")
    registrar(f"  pandas       {pandas.__version__}")
    registrar(f"  numpy        {numpy.__version__}")
    registrar(f"  joblib       {joblib.__version__}")
    try:
        import lightgbm
        registrar(f"  lightgbm     {lightgbm.__version__}")
    except ImportError:
        fallo("lightgbm no esta instalado (el modelo final lo necesita)")
    try:
        import streamlit
        registrar(f"  streamlit    {streamlit.__version__}")
    except ImportError:
        fallo("streamlit no esta instalado")
except Exception as e:
    fallo("No se pudo inspeccionar el entorno", e)

# ------------------------------------------------------------- artefactos
seccion("2. Carga de artefactos y coherencia de columnas")
ARTEFACTOS = None
try:
    import scoring_tfm
    with redirect_stderr(io.StringIO()):
        ARTEFACTOS = scoring_tfm.cargar_artefactos()
    ok("modelo/ carga sin errores")
    registrar(f"          modelo: {type(ARTEFACTOS['modelo']).__name__}")
    esperadas = scoring_tfm.columnas_esperadas(ARTEFACTOS)
    registrar(f"          columnas que espera el modelo: {len(esperadas)}")
    registrar(f"          {scoring_tfm.resumen_metricas(ARTEFACTOS)}")
    # Cada nombre declarado en el JSON corresponde a una clase concreta. No se puede comparar
    # por substring: "LightGBM" no aparece dentro de "LGBMClassifier".
    CLASES_ESPERADAS = {
        "lightgbm": {"LGBMClassifier"},
        "xgboost": {"XGBClassifier"},
        "histgradientboosting": {"HistGradientBoostingClassifier"},
        "randomforest": {"RandomForestClassifier"},
        "logisticregression": {"LogisticRegression"},
    }
    ganador = ARTEFACTOS.get("metricas", {}).get("modelo_ganador")
    clase_real = type(ARTEFACTOS["modelo"]).__name__
    esperadas = CLASES_ESPERADAS.get(str(ganador).lower().replace(" ", ""))
    if esperadas is None:
        registrar(f"  [----]  '{ganador}' no esta en la tabla de equivalencias; clase real: {clase_real}")
    elif clase_real in esperadas:
        ok(f"el modelo serializado ({clase_real}) coincide con lo declarado ('{ganador}')")
    else:
        fallo(f"El JSON dice '{ganador}' (se esperaba {esperadas}) pero el objeto es {clase_real}")
except Exception as e:
    fallo("No se pudieron cargar los artefactos", e)

# -------------------------------------------------------------- escenarios
seccion("3. Prediccion real (escenarios del Capitulo 6.2)")

ESCENARIOS = [
    ("Motociclista 22 anios, madrugada (3:00), atropello",
     dict(edad=22, tipo_vehiculo="Motocicleta", tipo_evento="Atropello", hora=3)),
    ("Automovil, 35 anios, tarde (15:00), colision",
     dict(edad=35, tipo_vehiculo="Automóvil", tipo_evento="Colisión", hora=15)),
]

if ARTEFACTOS is not None:
    try:
        D = ARTEFACTOS["diccionarios"]
        def cod(mapa_nombre, etiqueta):
            for c, e in D[mapa_nombre].items():
                if str(e).strip().lower() == etiqueta.strip().lower():
                    return c
            raise KeyError(f"'{etiqueta}' no esta en {mapa_nombre}")

        depto = cod("MAPA_DEPARTAMENTO", "Guatemala")
        muni = sorted(D.get("MUNICIPIOS_POR_DEPARTAMENTO", {}).get(depto, [])) or [None]
        muni = muni[0]
        sexo = cod("MAPA_SEXO", "Hombre")
        marca = "Toyota"
        registrar(f"  Parametros comunes: departamento=Guatemala, municipio={D['MAPA_MUNICIPIO'].get(muni, muni)}, "
                  f"sexo=Hombre, marca={marca}, mes=8, dia=15, dia_semana=5")
        registrar()

        resultados = {}
        for etiqueta, cfg in ESCENARIOS:
            X = scoring_tfm.construir_features(
                ARTEFACTOS, edad=cfg["edad"], sexo=sexo, departamento=depto, municipio=muni,
                tipo_vehiculo=cod("MAPA_TIPO_VEHICULO", cfg["tipo_vehiculo"]),
                tipo_evento=cod("MAPA_TIPO_EVENTO", cfg["tipo_evento"]),
                marca_vehiculo=marca, hora=cfg["hora"], mes=8, dia=15, dia_semana=5, anio=2026,
            )
            r = scoring_tfm.estimar_riesgo(ARTEFACTOS, X)
            resultados[etiqueta] = r["probabilidad"]
            registrar(f"  {etiqueta}")
            registrar(f"      -> {r['probabilidad']*100:.1f}%  ({r['nivel']})")
        ok("las predicciones corren de punta a punta")
        vals = list(resultados.values())
        if len(vals) == 2:
            registrar()
            registrar(f"  Diferencia entre escenarios: {(vals[0]-vals[1])*100:+.1f} puntos porcentuales")
            if vals[0] > vals[1]:
                ok("el escenario de mayor riesgo objetivo obtiene mayor probabilidad")
            else:
                fallo("el escenario de mayor riesgo NO obtiene mayor probabilidad -- revisar")
    except Exception as e:
        fallo("Fallo el calculo de escenarios", e)
else:
    registrar("  (omitido: los artefactos no cargaron)")

# ------------------------------------------------------------------- apps
def probar_app(nombre, pulsar_boton=None):
    seccion(f"{nombre}")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError as e:
        fallo("streamlit.testing no disponible (se necesita streamlit >= 1.28)", e)
        return
    try:
        at = AppTest.from_file(str(RAIZ / "apps" / nombre), default_timeout=120)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            at.run()
        if at.exception:
            for exc in at.exception:
                fallo(f"{nombre} lanzo una excepcion al arrancar: {exc.value}")
            return
        ok(f"{nombre} arranca sin excepciones")
        registrar(f"          widgets detectados: {len(at.selectbox)} selectores, "
                  f"{len(at.number_input)} numericos, {len(at.button)} botones")
        if at.error:
            for e_ in at.error:
                fallo(f"la app muestra st.error: {e_.value}")
        if pulsar_boton is not None and len(at.button) > pulsar_boton:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                at.button[pulsar_boton].click().run()
            if at.exception:
                for exc in at.exception:
                    fallo(f"{nombre} fallo al pulsar el boton: {exc.value}")
            else:
                ok("la accion principal se ejecuta sin excepciones")
                if at.metric:
                    for m in at.metric:
                        registrar(f"          metrica mostrada: {m.label} = {m.value}")
    except Exception as e:
        fallo(f"No se pudo ejecutar {nombre}", e)


probar_app("app.py", pulsar_boton=0)
probar_app("reporte_aseguradora.py", pulsar_boton=None)

# ----------------------------------------------------------------- resumen
seccion("RESUMEN")
if FALLOS == 0:
    registrar("  TODO EN ORDEN: las dos apps arrancan y el scoring funciona.")
else:
    registrar(f"  {FALLOS} problema(s) detectado(s). Ver el detalle arriba.")

with open(RAIZ / "tests" / "reporte_pruebas.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LINEAS) + "\n")
registrar()
registrar("Reporte guardado en reporte_pruebas.txt")
sys.exit(1 if FALLOS else 0)
