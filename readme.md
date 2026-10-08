# 📊 Proyecto de Minería de Datos: Análisis de Salud (Afiliados SIS)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-En%20Desarrollo-orange?style=for-the-badge)]()

Taller práctico de análisis y minería de datos enfocado en un dataset de salud pública de gran volumen (**más de 1.7 millones de registros**). El proyecto abarca desde la selección, limpieza y transformación de datos, hasta la aplicación de modelos de Machine Learning (Reglas de Asociación y Regresión) utilizando Python.

---

## 📑 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Guía de Instalación y Uso](#-guía-de-instalación-y-uso)
  - [1. Clonar el repositorio](#1-clonar-el-repositorio)
  - [2. Preparar los datos](#2-preparar-los-datos)
  - [3. Crear y activar entorno virtual](#3-crear-y-activar-un-entorno-virtual)
  - [4. Instalar dependencias](#4-instalar-dependencias)
  - [5. Ejecutar el análisis](#5-ejecutar-el-análisis)
- [Fases del Pipeline](#-fases-del-pipeline)
- [Tecnologías y Librerías](#-tecnologías-y-librerías)

---

## 🔬 Descripción General

El objetivo de este proyecto es extraer patrones de valor e inferencias a partir del registro de afiliados SIS:
- **Volumen**: Más de 1.7 millones de filas.
- **Técnicas aplicadas**:
  - Limpieza, imputación y tipificación de variables.
  - Análisis Exploratorio de Datos (EDA) y visualizaciones estadísticas.
  - Reglas de asociación (*Market Basket Analysis* / Apriori).
  - Modelos predictivos de regresión.

---

## 📁 Estructura del Proyecto

```text
mineros/
│
├── completo.py                    # Script principal con el pipeline de datos
├── .gitignore                     # Excluye datasets grandes (CSV) y entornos virtuales
├── requirements.txt               # Lista de dependencias del proyecto
└── readme.md                      # Documentación del proyecto
```

> [!IMPORTANT]
> **Dataset Excluido del Repositorio:**  
> El archivo `Afiliados_activos_DM_SIS.csv` (~340 MB) debe ubicarse en la raíz del proyecto. Está ignorado intencionalmente en `.gitignore` para no sobrepasar los límites de tamaño de GitHub.

---

## ⚙️ Requisitos Previos

Antes de comenzar, asegúrate de contar con:

- [Python 3.8+](https://www.python.org/downloads/) instalado en tu sistema.
- [Git](https://git-scm.com/) instalado y configurado.
- Una terminal de comandos (**PowerShell**, **CMD**, **Git Bash** o la terminal integrada de **VS Code**).

---

## 🚀 Guía de Instalación y Uso

### 1. Clonar el repositorio

Abre tu terminal y clona el proyecto en tu máquina local:

```bash
git clone https://github.com/OutFerz/mineros.git
cd mineros
```

---

### 2. Preparar los datos

1. Consigue o descarga el dataset `Afiliados_activos_DM_SIS.csv`.
2. Cópialo en la carpeta raíz del proyecto (al mismo nivel que `analisis_salud.py`):

mineros/
├── Afiliados_activos_DM_SIS.csv   <-- Colocar aquí
├── cantidad_poblacion.xlsx        <-- Colocar aquí (Datos INEI)
├── completo.py
...
```

> [!WARNING]
> No elimines ni modifiques la regla `*.csv` de tu `.gitignore` para evitar subir accidentalmente archivos pesados al repositorio remoto.

---

### 3. Crear y activar un entorno virtual

Es una buena práctica aislar las dependencias para evitar conflictos con otras librerías globales.

#### En Windows (PowerShell):
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

> *Si PowerShell muestra un error de políticas de ejecución de scripts, puedes habilitarlo temporalmente con:*  
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\env\Scripts\Activate.ps1
> ```

#### En Windows (CMD / Símbolo del sistema):
```cmd
python -m venv env
env\Scripts\activate.bat
```

#### En macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Instalar dependencias

Con el entorno virtual activado (`(env)` visible en la terminal), instala los paquetes requeridos:

```bash
pip install -r requirements.txt
```

> [!TIP]
> Si aún no has generado `requirements.txt`, puedes instalar las librerías base directamente:
> ```bash
> pip install pandas matplotlib seaborn scikit-learn mlxtend openpyxl
> ```

---

### 5. Ejecutar el análisis

Ejecuta el script principal para procesar los datos:

```bash
python completo.py
```

> [!NOTE]
> **Interacción y Resultados Automáticos:**  
> A lo largo de la ejecución, el script guardará todos los gráficos e informes generados (CSV, PNG) dentro de una nueva carpeta llamada `salidas/`. Además, el análisis se complementa con un documento HTML final interactivo.

---

## 🔄 Fases del Pipeline

```mermaid
flowchart LR
    A[Carga de Datos CSV] --> B[Limpieza y Preprocesamiento]
    B --> C[EDA y Estadísticas]
    C --> D[Visualización Gráfica]
    D --> E[Reglas de Asociación]
    E --> F[Modelos de Regresión]
```

1. **Selección y Carga:** Carga eficiente del archivo CSV.
2. **Preprocesamiento:** Manejo de nulos, tipificación y normalización.
3. **EDA:** Análisis descriptivo de variables clave.
4. **Visualización:** Gráficos y diagramas estadísticos.
5. **Modelado:** Extracción de patrones de asociación y modelos de regresión.

---

## 🛠️ Tecnologías y Librerías

| Herramienta | Uso Principal |
| :--- | :--- |
| **Python** | Lenguaje de programación base |
| **Pandas** | Manipulación y análisis de datos a gran escala |
| **Matplotlib** | Generación de gráficos y figuras |
| **Seaborn** | Visualizaciones estadísticas avanzadas |
| **Scikit-learn** | Modelado estadístico y Machine Learning |

---

## 👤 Autor

Desarrollado por [OutFerz](https://github.com/OutFerz).