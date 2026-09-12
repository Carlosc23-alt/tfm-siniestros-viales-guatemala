#!/usr/bin/env bash
# Prepara el historial de git para el repositorio del TFM.
#
# Cómo usarlo (en Windows, con Git for Windows instalado):
#   1. Abre la carpeta del proyecto en el explorador de Windows:
#      C:\Users\Carlos\Desktop\Maestria\TFM\tfm-siniestros-viales-guatemala
#   2. Copia este archivo (setup_git_repo.sh) dentro de esa carpeta.
#   3. Clic derecho dentro de la carpeta -> "Git Bash Here".
#   4. Ejecuta:  bash setup_git_repo.sh
#
# Si no tienes Git instalado, descárgalo primero de https://git-scm.com/download/win
# (incluye Git Bash, que es lo que necesitas para el paso 3).

set -e

if ! command -v git >/dev/null 2>&1; then
  echo "Git no está instalado. Descárgalo de https://git-scm.com/download/win e instálalo primero."
  exit 1
fi

if [ -d ".git" ]; then
  echo "Ya existe un repositorio git en esta carpeta (.git ya existe). No se hace nada."
  exit 0
fi

git init -b main
git config user.name "Carlos Alberto Calderón Illescas"
git config user.email "cacald01@ucm.es"

git add README.md LICENSE .gitignore requirements.txt requirements-dev.txt
git commit -m "chore: estructura inicial del proyecto" -m "Añade README, licencia MIT (con nota de atribución de datos INE/PNC), .gitignore y los archivos de dependencias (apps y entorno completo)."

git add data/
git commit -m "feat(data): agrega microdatos oficiales de siniestralidad vial (INE/PNC, 2018-2024)" -m "Incluye los 7 archivos .xlsx anuales publicados por el INE y el diccionario oficial de variables. Datos bajo licencia CC BY."

git add notebooks/ figuras/
git commit -m "feat(notebook): agrega EDA y modelado predictivo completo" -m "Notebook ejecutable de punta a punta: limpieza, análisis exploratorio, selección de modelo por validación cruzada 2018-2023, evaluación Out-of-Time sobre 2024 e interpretabilidad con SHAP. Incluye las figuras generadas del EDA y del análisis de interpretabilidad."

git add modelo/
git commit -m "feat(model): agrega artefactos del modelo final entrenado" -m "Modelo LightGBM serializado junto con el preprocesador, los diccionarios de codificación, las frecuencias y las métricas de evaluación sobre el holdout Out-of-Time de 2024."

git add src/
git commit -m "feat(src): agrega capa de scoring, descarga y reentrenamiento automatizado" -m "scoring_tfm.py centraliza la ingeniería de características que consumen ambas apps. descargar_datos.py automatiza la descarga desde la API CKAN del INE. reentrenar_modelo.py reentrena con una compuerta de aprobación humana antes de promover un modelo nuevo."

git add apps/
git commit -m "feat(apps): agrega apps de Streamlit para simulación y triage" -m "app.py: simulador de riesgo de fatalidad para un siniestro dado. reporte_aseguradora.py: prototipo de captura en campo con triage para uso de una aseguradora."

git add tests/
git commit -m "test: agrega prueba de humo para validar modelo y apps" -m "Verifica que el modelo cargue correctamente, que las columnas coincidan con lo esperado y que ambas apps arranquen sin errores."

echo ""
echo "Listo. Historial de commits creado:"
git log --oneline --reverse

echo ""
echo "Siguiente paso: crea el repositorio vacio en GitHub y conectalo."
echo "  1. Ve a https://github.com/new"
echo "  2. Nombre: tfm-siniestros-viales-guatemala | Visibilidad: Publico | NO marques README/.gitignore/licencia"
echo "  3. Copia la URL del repo y ejecuta:"
echo "     git remote add origin https://github.com/TU_USUARIO/tfm-siniestros-viales-guatemala.git"
echo "     git push -u origin main"
