# Modelado predictivo de la severidad y riesgo de fatalidad en siniestros viales en Guatemala

Trabajo Fin de Máster — Máster en Big Data, Data Science e Inteligencia Artificial
Universidad Complutense de Madrid · Carlos Alberto Calderón Illescas · 2026

---

## Qué hace este proyecto

Estima, a partir de los microdatos oficiales de siniestralidad vial de Guatemala
(INE/PNC, 2018–2024), la **probabilidad de que una persona involucrada en un siniestro
resulte fallecida** en lugar de lesionada.

El modelo final es un **LightGBM** entrenado con 2018–2023 y evaluado sobre un holdout
*Out-of-Time* de 2024 que nunca se usó para seleccionar nada:

| Métrica (holdout 2024, 11,272 casos) | Valor |
|---|---|
| ROC-AUC | **0.740** |
| PR-AUC | 0.469 |
| Recall (clase fallecido) | 67.8% |
| Precisión (clase fallecido) | 35.7% |
| F1 (clase fallecido) | 0.467 |

Tasa base de fatalidad del periodo: **19.1%**.

---

## Demo en vivo

Las dos aplicaciones están desplegadas en Streamlit Community Cloud y se pueden abrir sin
instalar nada:

- **Simulador de riesgo** — [tfm-siniestros-viales-guatemala-estimador.streamlit.app](https://tfm-siniestros-viales-guatemala-estimador.streamlit.app/)
- **Reporte y triage en campo** — [tfm-siniestros-viales-guatemala-reporte.streamlit.app](https://tfm-siniestros-viales-guatemala-reporte.streamlit.app/)

Los reportes que se envían en la segunda no se conservan: corre en un contenedor efímero y es
una prueba de concepto. La propia aplicación lo advierte.

---

## Estructura del repositorio

```
.
├── notebooks/
│   └── TFM_Analisis_Completo.ipynb   Análisis completo, ejecutable de punta a punta
├── src/
│   ├── scoring_tfm.py                Capa de scoring compartida por las dos apps
│   ├── descargar_datos.py            Descarga automática desde la API CKAN del INE
│   └── reentrenar_modelo.py          Reentrenamiento con compuerta de aprobación humana
├── apps/
│   ├── app.py                        Simulador de riesgo (Streamlit)
│   └── reporte_aseguradora.py        Prototipo de captura en campo + triage
├── tests/
│   ├── probar_apps.py                Prueba de humo de ambas apps
│   └── reporte_pruebas.txt           Última salida de esa prueba
├── modelo/                           Artefactos serializados del modelo entrenado
├── requirements.txt                  Dependencias de las apps (lo que instala el deploy)
├── requirements-dev.txt              Entorno completo para reejecutar el notebook
├── data/
│   ├── raw/                          Los 7 .xlsx anuales publicados por el INE
│   └── diccionario-...xlsx           Diccionario oficial de variables
└── figuras/                          Figuras del EDA y de interpretabilidad
```

---

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
pip install -r requirements-dev.txt   # entorno completo (notebook + apps)
pip install -r requirements.txt       # solo lo necesario para las apps
```

**Verificar que todo está en su sitio** (carga el modelo, comprueba que las columnas
coinciden y arranca las dos apps sin ejecutar el notebook):

```bash
python tests/probar_apps.py
```

**Simulador de riesgo:**

```bash
streamlit run apps/app.py
```

**Prototipo de captura y triage:**

```bash
streamlit run apps/reporte_aseguradora.py
```

**Rehacer el análisis completo:** abrir `notebooks/TFM_Analisis_Completo.ipynb` y ejecutarlo
de principio a fin. La primera celda sitúa el directorio de trabajo en la raíz del repositorio,
de modo que todas las rutas relativas funcionan igual desde `notebooks/` o desde la raíz.

**Descargar una copia fresca de los datos del INE:**

```bash
python src/descargar_datos.py --carpeta ./data/verificacion/
```

**Reentrenar cuando el INE publique un año nuevo:**

```bash
python src/reentrenar_modelo.py            # descarga, reentrena, compara y NO promueve
python src/reentrenar_modelo.py --promover # promueve solo si el ROC-AUC no empeora
```

---

## Notas metodológicas

- **Partición Out-of-Time.** La selección de modelo se fijó por validación cruzada sobre
  2018–2023 *antes* de mirar el holdout de 2024. HistGradientBoosting obtiene 0.743 en ese
  holdout frente al 0.740 de LightGBM, y aun así se mantuvo LightGBM: cambiar de ganador
  después de ver el holdout lo convertiría en un conjunto de validación más.
- **Desbalanceo.** 80.9% lesionados / 19.1% fallecidos. Se trató con `sample_weight` en los
  modelos de boosting y `class_weight="balanced"` en la regresión logística y el random forest.
- **Interpretabilidad.** Los valores SHAP se calculan sobre el holdout completo (11,272 casos),
  no sobre una submuestra. `figuras/06b_*` documenta la comprobación de estabilidad.
- **Una sola copia de la lógica de scoring.** `src/scoring_tfm.py` es la única implementación
  de la ingeniería de características, y lee las columnas que el modelo espera desde
  `modelo/features_tfm.json`. Las apps la importan; no reimplementan nada.

---

## Datos y licencia

El **código** de este repositorio se publica bajo licencia MIT (ver `LICENSE`).

Los **datos** de `data/` proceden del portal de datos abiertos del Instituto Nacional de
Estadística de Guatemala, publicados bajo **Creative Commons Attribution (CC BY)**, que
permite su uso, redistribución y transformación citando la fuente. Los registros no
contienen identificadores personales.

Fuente: INE y PNC, *Hechos de tránsito con personas fallecidas y lesionadas, 2018–2024*.
[datos.ine.gob.gt](https://datos.ine.gob.gt/en/dataset/accidentes-de-transito-fallecidos-y-lesionados)

---

## Limitaciones

Los microdatos son de naturaleza **administrativa**: describen el registro del hecho (cuándo,
dónde, qué vehículo, qué tipo de evento) pero no miden ningún mecanismo causal de la severidad
— no hay velocidad, ni uso de casco o cinturón, ni estado de la vía, ni alcoholemia, ni tiempo
de respuesta del auxilio. El techo de desempeño lo fijan las variables disponibles, no el
algoritmo. El Capítulo 7.4 de la memoria propone qué debería capturarse para superarlo.
