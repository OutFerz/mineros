Proyecto de Minería de Datos: Análisis de Salud (Afiliados SIS)

Este proyecto es un taller de análisis de datos enfocado en un dataset de salud de gran volumen (más de 1.7 millones de registros). Incluye procesos de selección, limpieza, transformación de datos y la aplicación de modelos de Machine Learning (Reglas de Asociación y Regresión) utilizando Python.

Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

📁 mineros/
│
├── analisis_salud.py              # Script principal con el pipeline de datos
├── .gitignore                     # Archivo que evita la subida de los datasets (CSV)
├── requirements.txt               # Lista de dependencias y librerías de Python
└── README.md                      # Documentación del proyecto


Nota: El archivo Afiliados_activos_DM_SIS.csv debe colocarse en la raíz del proyecto, pero está excluido del control de versiones mediante el .gitignore debido a su tamaño.

Requisitos Previos

Python 3.8 o superior instalado en el sistema.

Git instalado.

Terminal o consola de comandos (PowerShell, CMD, Git Bash o la consola de VS Code).

Paso a Paso: Instalación y Uso

1. Clonar el repositorio

Si estás descargando este proyecto desde GitHub, abre tu terminal y ejecuta:

git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd Taller_Salud

2. Preparar los datos

Descarga el dataset original Afiliados_activos_DM_SIS.csv y colócalo en la carpeta raíz del proyecto (junto a este README).

Asegúrate de no forzar su subida a Git.

3. Crear y activar un entorno virtual

Es una buena práctica aislar las dependencias del proyecto.

En Windows:

python -m venv venv
.\venv\Scripts\activate

En Mac / Linux:

python3 -m venv venv
source venv/bin/activate

4. Instalar las dependencias

Con el entorno virtual activado, instala las librerías necesarias ejecutando:

pip install -r requirements.txt

5. Ejecutar el análisis

Para correr el pipeline completo (Limpieza, Estadísticos, Reglas de Asociación y Regresión), ejecuta:

python analisis_salud.py

Nota: A lo largo de la ejecución, se abrirán ventanas emergentes con gráficos (Matplotlib/Seaborn). Debes cerrar cada ventana gráfica para que el script continúe con el siguiente paso.