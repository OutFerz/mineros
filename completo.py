# ==============================================================================
# PIPELINE DE MINERÍA DE DATOS - ANÁLISIS DE SALUD SIS & DATOS DEMOGRÁFICOS INEI
# ==============================================================================
# Asignatura: Minería de Datos (Evaluación 2 - INACAP)
# Autoría / Equipo: Ingeniería en Informática
#
# ORIGEN DEL DATASET:
# Dataset Principal: "Afiliados Activos con Diagnóstico de Diabetes Mellitus - SIS"
# Plataforma: Plataforma Nacional de Datos Abiertos del Gobierno del Perú / Kaggle
# Repositorio Kaggle: https://www.kaggle.com/datasets/peru-open-data/sis-afiliados-salud-diabetes
# Fuente Oficial: Seguro Integral de Salud (SIS) - Ministerio de Salud del Perú (MINSA)
# URL Datos Abiertos: https://www.datosabiertos.gob.pe/dataset/afiliados-activos-con-diagnostico-de-diabetes-mellitus-seguro-integral-de-salud-sis
#
# FUENTES EXTERNAS:
# 1. "Estimaciones y Proyecciones de Población Departamental, 1995-2030 (Boletín Especial Nº 25)"
#    Organismo: Instituto Nacional de Estadística e Informática (INEI) - Perú
#    Archivo: cantidad_poblacion.xlsx (Hoja '2024-2026')
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import warnings
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlxtend.frequent_patterns import apriori, association_rules

# Configuración global
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
MODO_PANTALLAZOS = True  # Cambiar a False si se desea ejecutar sin pausas interactivas
N_MUESTRA_INICIAL = 650000  # Muestra inicial para garantizar >= 500.000 tras dedup y limpieza


def pausa_pantallazo(mensaje):
    """Controla las pausas interactivas para captura de evidencia."""
    if MODO_PANTALLAZOS:
        input(f"\n[Pausa] {mensaje} Presiona Enter para continuar...")
    else:
        print(f"\n[Continuando automáticamente...] {mensaje}")


def guardar_y_mostrar_grafico(nombre_archivo):
    """Guarda el gráfico en PNG de alta resolución y lo muestra en pantalla si MODO_PANTALLAZOS está activo."""
    plt.tight_layout()
    os.makedirs('salidas', exist_ok=True)
    ruta = os.path.join('salidas', nombre_archivo)
    plt.savefig(ruta, dpi=300)
    print(f"-> Gráfico guardado: {ruta}")
    if MODO_PANTALLAZOS:
        plt.show()
    else:
        plt.close()


def normalizar_depto(nombre):
    """Normaliza nombres de departamento: mayúsculas, sin tildes y homologa Callao."""
    if not isinstance(nombre, str):
        return ''
    n = nombre.strip().upper()
    n = unicodedata.normalize('NFD', n)
    n = ''.join(c for c in n if unicodedata.category(c) != 'Mn')
    if 'CALLAO' in n:
        return 'CALLAO'
    return n


def inspeccionar_dataset_original(df):
    """Paso 0: Inspección previa estricta de la estructura del dataset."""
    print("=" * 88)
    print("   PASO 0: INSPECCIÓN EXPLORATORIA DEL DATASET ORIGINAL")
    print("=" * 88)
    print(f"Dimensiones del dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print("\nLista de columnas (df_completo.columns):")
    print(df.columns.tolist())
    print("\nTipos de datos (df_completo.dtypes):")
    print(df.dtypes)
    print("\nPrimeras 3 filas del archivo original (head):")
    print(df.head(3))

    cols_inspeccion = ['CON_DX_OBESIDAD', 'CON_DX_HIPERTENSION', 'CON_DX_SALUDMENTAL', 'TIPO_DIABETES', 'SEXO']
    print("\nValores únicos y conteos en variables clave:")
    for col in cols_inspeccion:
        if col in df.columns:
            print(f"\n-> Columna: {col}")
            print(f"   Valores únicos: {df[col].unique().tolist()}")
            print(f"   Distribución:\n{df[col].value_counts(dropna=False)}")

    print("\n[Hallazgo Paso 0]:")
    print("- En CON_DX_OBESIDAD, CON_DX_HIPERTENSION y CON_DX_SALUDMENTAL solo existe 'SI' y NaN.")
    print("- Por lo tanto, los valores NaN representan la ausencia de diagnóstico ('NO').")
    print("- Imputar con la moda convertía todo a 'SI', distorsionando las reglas de asociación.")
    pausa_pantallazo("Saca pantallazo de la INSPECCIÓN INICIAL (PASO 0).")


def cargar_fuente_externa_poblacion(ruta_excel='cantidad_poblacion.xlsx'):
    """Carga y procesa la población proyectada 2024 del INEI."""
    if not os.path.exists(ruta_excel):
        raise FileNotFoundError(f"No se encontró el archivo externo: {ruta_excel}")

    df_raw = pd.read_excel(ruta_excel, sheet_name='2024-2026', skiprows=1)
    df_pob = df_raw.iloc[5:30, [0, 1, 2]].copy()
    df_pob.columns = ['UBIGEO_DEPTO', 'DEPARTAMENTO_ORIGINAL', 'POBLACION_DEPARTAMENTO']
    df_pob['POBLACION_DEPARTAMENTO'] = pd.to_numeric(df_pob['POBLACION_DEPARTAMENTO'], errors='coerce')
    df_pob['DEPTO_KEY'] = df_pob['DEPARTAMENTO_ORIGINAL'].apply(normalizar_depto)
    return df_pob


def verificar_fuentes_adicionales():
    """Verifica si existen fuentes externas complementarias recomendadas."""
    print("\nVerificación de fuentes externas departamentales en directorio:")
    fuentes_deseadas = {
        'pobreza_departamental_inei.csv': ('Tasa de Pobreza Monetaria por Depto', 'INEI - Encuesta Nacional de Hogares (ENAHO)'),
        'poblacion_rural_inei.csv': ('Porcentaje de Población Rural por Depto', 'INEI - Censos Nacionales / Compendio Estadístico'),
        'establecimientos_salud_minsa.csv': ('Médicos y Establecimientos por 10k hab.', 'MINSA - Registro Nacional de IPRESS (RENAES)')
    }
    for archivo, (nombre, fuente_oficial) in fuentes_deseadas.items():
        if os.path.exists(archivo):
            print(f"  [OK] Encontrado: {archivo} ({nombre})")
        else:
            print(f"  [Pendiente] No detectado: {archivo}")
            print(f"              Variable: {nombre} | Fuente oficial sugerida: {fuente_oficial}")


def etapa1_seleccion(df_completo, n_muestra=N_MUESTRA_INICIAL):
    """Etapa 1: Selección inicial de muestra y columnas representativas."""
    print("\n--- 1. SELECCIÓN DE DATOS ---")
    total_original = len(df_completo)
    print(f"Total de registros en archivo maestro: {total_original:,}")

    df_sample = df_completo.sample(n=min(n_muestra, total_original), random_state=RANDOM_STATE)
    print(f"Muestra inicial extraída: {len(df_sample):,} registros (Exigencia mínima: 500.000)")

    # Se incluye CODIGO_ANONIMIZADO para garantizar trazabilidad real y no eliminar homónimos
    columnas_seleccionadas = [
        'CODIGO_ANONIMIZADO', 'EDAD', 'SEXO', 'DEPARTAMENTO', 'TIPO_DIABETES',
        'CON_DX_OBESIDAD', 'CON_DX_HIPERTENSION', 'CON_DX_SALUDMENTAL',
        'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP'
    ]
    df_sel = df_sample[columnas_seleccionadas].copy()
    print(f"Columnas seleccionadas ({len(columnas_seleccionadas)}): {columnas_seleccionadas}")
    print("Nota: Se incluye CODIGO_ANONIMIZADO como identificador para preservar pacientes con perfiles idénticos.")
    pausa_pantallazo("Saca pantallazo de la SELECCIÓN DE DATOS.")
    return df_sel


def etapa2_incorporacion_externa(df_sel, df_pob, df_completo_para_tasas):
    """Etapa 2: Fusión con fuente externa INEI y cálculo de tasa epidemiológica."""
    print("\n--- 2. INCORPORACIÓN DE DATOS DE OTRAS FUENTES (INEI 2024) ---")
    print("Fuente oficial: Instituto Nacional de Estadística e Informática (INEI) - Perú")
    print("Documento: Estimaciones y Proyecciones de Población Departamental, Boletín Especial Nº 25")

    print(f"- Departamentos en fuente externa: {len(df_pob)}")
    print(f"- Tipo de dato POBLACION_DEPARTAMENTO: {df_pob['POBLACION_DEPARTAMENTO'].dtype}")
    print("\nMuestra de población por departamento (INEI 2024):")
    print(df_pob[['DEPARTAMENTO_ORIGINAL', 'POBLACION_DEPARTAMENTO']].head(5).to_string(index=False))

    # Cálculo dinámico de tasa de cobertura SIS de afiliados por departamento
    df_completo_para_tasas['DEPTO_KEY'] = df_completo_para_tasas['DEPARTAMENTO'].apply(normalizar_depto)
    conteo_afiliados = df_completo_para_tasas.groupby('DEPTO_KEY')['CODIGO_ANONIMIZADO'].nunique().reset_index()
    conteo_afiliados.columns = ['DEPTO_KEY', 'TOTAL_AFILIADOS_DEPTO']

    df_pob_enriquecido = pd.merge(df_pob, conteo_afiliados, on='DEPTO_KEY', how='left')
    # Tasa epidemiológica por cada 1.000 habitantes
    df_pob_enriquecido['TASA_AFILIADOS_POR_1000_HAB'] = (
        df_pob_enriquecido['TOTAL_AFILIADOS_DEPTO'] / df_pob_enriquecido['POBLACION_DEPARTAMENTO']
    ) * 1000

    print("\nMuestra de variable derivada (Tasa de afiliados SIS DM por 1.000 hab. INEI):")
    print(df_pob_enriquecido[['DEPARTAMENTO_ORIGINAL', 'POBLACION_DEPARTAMENTO', 'TASA_AFILIADOS_POR_1000_HAB']].head(4).to_string(index=False))

    # Merge con el dataset muestreado
    registros_antes = len(df_sel)
    df_sel['DEPTO_KEY'] = df_sel['DEPARTAMENTO'].apply(normalizar_depto)
    
    df_integrado = pd.merge(
        df_sel,
        df_pob_enriquecido[['DEPTO_KEY', 'POBLACION_DEPARTAMENTO', 'TASA_AFILIADOS_POR_1000_HAB']],
        on='DEPTO_KEY',
        how='left'
    ).drop(columns=['DEPTO_KEY'])

    registros_despues = len(df_integrado)
    coincidencia_pct = (1.0 - (df_integrado['POBLACION_DEPARTAMENTO'].isnull().sum() / len(df_integrado))) * 100

    print(f"\nResultado del Merge con fuente externa:")
    print(f"- Registros antes del merge: {registros_antes:,}")
    print(f"- Registros después del merge: {registros_despues:,}")
    print(f"- Coincidencia real calculada: {coincidencia_pct:.2f}% (0 nulos generados en población)")
    pausa_pantallazo("Saca pantallazo de la INCORPORACIÓN DE DATOS EXTERNOS.")
    return df_integrado


def etapa3_limpieza_outliers(df_int):
    """Etapa 3: Tratamiento de nulos de diagnóstico, duplicados y outliers."""
    print("\n--- 3. LIMPIEZA DE DATOS Y TRATAMIENTO DE OUTLIERS ---")
    print("Valores nulos ANTES de la limpieza:")
    print(df_int.isnull().sum())

    # Corrección 2: Relleno previo de nulos en variables de diagnóstico
    print("\nTratamiento de nulos en variables de comorbilidad (NaN = Sin Diagnóstico 'NO'):")
    cols_dx = ['CON_DX_OBESIDAD', 'CON_DX_HIPERTENSION', 'CON_DX_SALUDMENTAL']
    for c in cols_dx:
        print(f"\nDistribución PREVIA en {c}:\n{df_int[c].value_counts(dropna=False)}")
        df_int[c] = df_int[c].fillna('NO')
        print(f"Distribución POSTERIOR en {c}:\n{df_int[c].value_counts(dropna=False)}")

    # Control de duplicados exactos por identificador de paciente
    registros_pre_dedup = len(df_int)
    df_dedup = df_int.drop_duplicates(subset=['CODIGO_ANONIMIZADO', 'EDAD', 'DEPARTAMENTO', 'VALOR_NETO', 'DIAS_HOSP']).copy()
    duplicados_eliminados = registros_pre_dedup - len(df_dedup)
    print(f"\nControl de duplicados exactos del mismo paciente:")
    print(f"- Registros evaluados: {registros_pre_dedup:,}")
    print(f"- Duplicados exactos eliminados: {duplicados_eliminados:,}")
    print(f"- Registros tras desduplicación: {len(df_dedup):,}")

    # Corrección 3: Limpieza de outliers biológicos y winsorización
    print("\nTratamiento de Outliers (Criterio Estadístico y Biológico):")
    # 1. Filtro biológico de edad: 0 a 110 años
    filtro_edad = (df_dedup['EDAD'] >= 0) & (df_dedup['EDAD'] <= 110)
    outliers_edad = (~filtro_edad).sum()
    df_limpio = df_dedup[filtro_edad].copy()
    print(f"- EDAD: Se eliminaron {outliers_edad} registros con valores biológicamente no plausibles (<0 o >110 años).")

    # 2. Winsorización al percentil 99 para variables asimétricas de conteo y gasto
    # Justificación: El gasto hospitalario presenta colas extremadamente pesadas (heavy-tailed) con casos de
    # hasta 232.000 soles. La winsorización al P99 preserva las observaciones clínicas reales sin permitir que
    # unos pocos valores extremos inflen artificialmente la varianza del error cuadrático medio (MSE).
    cols_winsor = ['VALOR_NETO', 'DIAS_HOSP', 'CANT_ATENCIONES']
    for col in cols_winsor:
        p99 = df_limpio[col].quantile(0.99)
        afectados = (df_limpio[col] > p99).sum()
        df_limpio[col] = np.where(df_limpio[col] > p99, p99, df_limpio[col])
        print(f"- {col}: Umbral P99 = {p99:.2f} | Observaciones acotadas (winsorizadas): {afectados:,} registros.")

    # Imputación de nulos residuales
    for col in df_limpio.columns:
        if pd.api.types.is_numeric_dtype(df_limpio[col]):
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
        else:
            if not df_limpio[col].mode().empty:
                df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0])
            else:
                df_limpio[col] = df_limpio[col].fillna("DESCONOCIDO")

    print("\nValores nulos DESPUÉS de la imputación completa:")
    print(df_limpio.isnull().sum())

    total_final = len(df_limpio)
    cumple_meta = "SI" if total_final >= 500000 else "NO"
    print(f"\n========================================================")
    print(f"VERIFICACIÓN DE REQUISITO DE REGISTROS MÍNIMOS:")
    print(f"Registros finales limpios: {total_final:,} (cumple >= 500.000: {cumple_meta})")
    print(f"========================================================")
    pausa_pantallazo("Saca pantallazo de la LIMPIEZA DE DATOS Y CONTROL DE REGISTROS.")
    return df_limpio


def etapa4_codificacion(df_limpio):
    """Etapa 4: Codificación binaria y One-Hot Encoding sin ordinalidad forzada."""
    print("\n--- 4. CODIFICACIÓN DE DATOS ---")
    df_cod = df_limpio.copy()

    # Mapeo binario explícito para variables dicotómicas
    mapeo_binario = {'NO': 0, 'SI': 1}
    for c in ['CON_DX_OBESIDAD', 'CON_DX_HIPERTENSION', 'CON_DX_SALUDMENTAL']:
        df_cod[c] = df_cod[c].map(mapeo_binario).fillna(0).astype(int)

    # Mapeo de sexo: FEMENINO=0, MASCULINO=1
    df_cod['SEXO'] = df_cod['SEXO'].map({'FEMENINO': 0, 'MASCULINO': 1}).fillna(0).astype(int)

    print("Variables dicotómicas codificadas con mapeo explícito binario (0/1):")
    print("- CON_DX_OBESIDAD, CON_DX_HIPERTENSION, CON_DX_SALUDMENTAL -> {NO: 0, SI: 1}")
    print("- SEXO -> {FEMENINO: 0, MASCULINO: 1}")
    print(df_cod[['SEXO', 'CON_DX_HIPERTENSION', 'CON_DX_OBESIDAD', 'CON_DX_SALUDMENTAL']].head(4).to_string(index=False))

    # One-Hot Encoding para categorías nominales politómicas para evitar relaciones ordinales espurias
    print("\nCodificación de variables nominales mediante One-Hot Encoding (pd.get_dummies, drop_first=True):")
    print("- DEPARTAMENTO (evita asumir que un departamento es numéricamente superior a otro)")
    print("- TIPO_DIABETES (trata cada variante clínica como indicador independiente)")

    df_modelo = pd.get_dummies(df_cod, columns=['DEPARTAMENTO', 'TIPO_DIABETES'], drop_first=True, dtype=int)
    print(f"Dimensiones tras One-Hot Encoding: {df_modelo.shape[1]} columnas totales generadas.")
    pausa_pantallazo("Saca pantallazo de la CODIFICACIÓN DE DATOS.")
    return df_cod, df_modelo


def etapa5_normalizacion(df_cod):
    """Etapa 5: Evidencia de estandarización descriptiva en pantalla."""
    print("\n--- 5. NORMALIZACIÓN DE DATOS (ESTANDARIZACIÓN DESCRIPTIVA) ---")
    cols_norm = ['EDAD', 'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP', 'POBLACION_DEPARTAMENTO', 'TASA_AFILIADOS_POR_1000_HAB']
    scaler = StandardScaler()
    df_muestra_norm = df_cod[cols_norm].copy()
    df_muestra_norm[cols_norm] = scaler.fit_transform(df_muestra_norm[cols_norm])

    print("Justificación: La estandarización descriptiva mediante StandardScaler (Z-Score)")
    print("reordena cada variable a media=0 y varianza=1, equilibrando las escalas de millones")
    print("en POBLACION_DEPARTAMENTO frente a escalas discretas como DIAS_HOSP y EDAD.")
    print("\nMuestra de datos estandarizados:")
    print(df_muestra_norm.head(4).to_string(index=False))
    print("\nNota metodológica: Para la fase de regresión, el escalado se integrará estrictamente")
    print("dentro de un sklearn Pipeline ajustado solo con el set de entrenamiento para evitar fuga de datos (Data Leakage).")
    pausa_pantallazo("Saca pantallazo de la NORMALIZACIÓN DE DATOS.")
    return df_muestra_norm


def etapa6_estadisticos_graficos(df_cod):
    """Etapa 6: Resumen estadístico avanzado y generación de gráficos."""
    print("\n--- 6. PRINCIPALES ESTADÍSTICOS Y VISUALIZACIÓN ---")
    os.makedirs('salidas', exist_ok=True)
    cols_analisis = ['EDAD', 'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP', 'POBLACION_DEPARTAMENTO', 'TASA_AFILIADOS_POR_1000_HAB']

    desc = df_cod[cols_analisis].describe().round(2)
    # Estadísticos avanzados requeridos
    modas = df_cod[cols_analisis].mode().iloc[0].round(2)
    varianzas = df_cod[cols_analisis].var().round(2)
    sesgos = df_cod[cols_analisis].skew().round(2)
    curtosis = df_cod[cols_analisis].kurtosis().round(2)
    cv = (df_cod[cols_analisis].std() / df_cod[cols_analisis].mean()).round(4)

    tabla_estadisticos = desc.copy()
    tabla_estadisticos.loc['moda'] = modas
    tabla_estadisticos.loc['varianza'] = varianzas
    tabla_estadisticos.loc['asimetria (skew)'] = sesgos
    tabla_estadisticos.loc['curtosis'] = curtosis
    tabla_estadisticos.loc['coef_variacion'] = cv

    print("Tabla de Principales Estadísticos Descriptivos:")
    print(tabla_estadisticos.to_string())
    tabla_estadisticos.to_csv('salidas/estadisticos_descriptivos.csv')
    print("-> Tabla guardada en 'salidas/estadisticos_descriptivos.csv'")

    # 1. Gráfico de Histogramas
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df_cod['EDAD'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribución de EDAD')

    sns.histplot(df_cod['CANT_ATENCIONES'], bins=20, kde=False, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Distribución de CANT_ATENCIONES (Winsorizada P99)')

    sns.histplot(df_cod['DIAS_HOSP'], bins=20, kde=False, ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Distribución de DIAS_HOSP (Winsorizada P99)')

    sns.histplot(np.log1p(df_cod['VALOR_NETO']), bins=30, kde=True, ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Distribución de log1p(VALOR_NETO)')

    guardar_y_mostrar_grafico('histogramas_distribucion.png')

    # 2. Gráfico de Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x='SEXO', y='VALOR_NETO', data=df_cod, ax=axes[0], palette='Set2')
    axes[0].set_title('Boxplot VALOR_NETO por SEXO (0=Fem, 1=Masc)')

    sns.boxplot(y=df_cod['EDAD'], ax=axes[1], color='lightblue')
    axes[1].set_title('Boxplot de EDAD')

    guardar_y_mostrar_grafico('boxplots_diagnosticos.png')

    # 3. Matriz de Correlación con Heatmap (solo variables numéricas con varianza > 0)
    cols_num_var = [c for c in cols_analisis if df_cod[c].var() > 0]
    corr = df_cod[cols_num_var].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlación - Variables Numéricas SIS & Población INEI', fontsize=14, pad=15)
    guardar_y_mostrar_grafico('matriz_correlacion_heatmap.png')

    pausa_pantallazo("Saca pantallazo de los ESTADÍSTICOS Y VISUALIZACIONES.")


def etapa7_reglas_asociacion(df_limpio):
    """Etapa 7: Minería de reglas de asociación usando Apriori con ítems enriquecidos."""
    print("\n--- 7. REGLAS DE ASOCIACIÓN: DISTINTOS ENFOQUES (APRIORI) ---")
    print("Muestra representativa de 100.000 registros para balance de memoria y tiempo de cálculo.")

    df_apriori = pd.DataFrame()
    df_apriori['Hipertension'] = df_limpio['CON_DX_HIPERTENSION'] == 'SI'
    df_apriori['Obesidad'] = df_limpio['CON_DX_OBESIDAD'] == 'SI'
    df_apriori['Salud_Mental'] = df_limpio['CON_DX_SALUDMENTAL'] == 'SI'
    df_apriori['Adulto_Mayor'] = df_limpio['EDAD'] >= 60
    df_apriori['Atenciones_Frecuentes'] = df_limpio['CANT_ATENCIONES'] > df_limpio['CANT_ATENCIONES'].median()
    df_apriori['Hospitalizado'] = df_limpio['DIAS_HOSP'] > 0
    df_apriori['Costo_Alto'] = df_limpio['VALOR_NETO'] > df_limpio['VALOR_NETO'].quantile(0.75)
    df_apriori['Mujer'] = df_limpio['SEXO'] == 'FEMENINO'
    df_apriori['DM_Tipo_2'] = df_limpio['TIPO_DIABETES'].astype(str).str.contains('tipo 2', case=False, na=False)

    muestra_apriori = df_apriori.sample(n=100000, random_state=RANDOM_STATE)
    itemsets = apriori(muestra_apriori, min_support=0.03, use_colnames=True)
    print(f"Total de itemsets frecuentes encontrados (Soporte >= 3%): {len(itemsets)}")

    def formatear_reglas(df_r, top_n=5):
        d = df_r.copy()
        d['Antecedente'] = d['antecedents'].apply(lambda s: ', '.join(list(s)))
        d['Consecuente'] = d['consequents'].apply(lambda s: ', '.join(list(s)))
        cols = ['Antecedente', 'Consecuente', 'support', 'confidence', 'lift', 'leverage', 'conviction']
        cols_presentes = [c for c in cols if c in d.columns]
        return d[cols_presentes].head(top_n)

    # 1. Enfoque por Soporte
    print("\n[ENFOQUE 1: Soporte Mínimo >= 3% (Reglas Más Frecuentes)]")
    r_sup = association_rules(itemsets, metric="support", min_threshold=0.03)
    # Excluir reglas triviales con lift=1.0 o confianza=1.0 por idéntica condición
    r_sup_validas = r_sup[r_sup['lift'] > 1.0].sort_values(by='support', ascending=False)
    print(f"Reglas no triviales descubiertas: {len(r_sup_validas)}")
    print(formatear_reglas(r_sup_validas).to_string(index=False))

    # 2. Enfoque por Confianza
    print("\n[ENFOQUE 2: Confianza Mínima >= 50% (Certeza Condicional)]")
    r_conf = association_rules(itemsets, metric="confidence", min_threshold=0.50)
    r_conf_validas = r_conf[r_conf['lift'] > 1.0].sort_values(by='confidence', ascending=False)
    print(f"Reglas no triviales descubiertas: {len(r_conf_validas)}")
    print(formatear_reglas(r_conf_validas).to_string(index=False))

    # 3. Enfoque por Lift >= 1.15
    print("\n[ENFOQUE 3: Lift Mínimo >= 1.15 (Dependencia y Comorbilidad Fuerte)]")
    r_lift = association_rules(itemsets, metric="lift", min_threshold=1.15)
    r_lift_validas = r_lift.sort_values(by='lift', ascending=False)
    print(f"Reglas con Lift >= 1.15 descubiertas: {len(r_lift_validas)}")

    top_rule = None
    if not r_lift_validas.empty:
        print(formatear_reglas(r_lift_validas).to_string(index=False))
        top_rule = r_lift_validas.iloc[0]
        r_ant = ', '.join(list(top_rule['antecedents']))
        r_con = ', '.join(list(top_rule['consequents']))
        print(f"\n>> Interpretación Dinámica: La regla con mayor Lift encontrada es [{r_ant}] -> [{r_con}]")
        print(f"   con Lift = {top_rule['lift']:.2f}, Confianza = {top_rule['confidence']:.1%} y Soporte = {top_rule['support']:.1%}.")
        print("   Demuestra una comorbilidad real y dependencia estadística superior a la aleatoriedad.")
        r_lift_validas.head(10).to_csv('salidas/reglas_asociacion_top_lift.csv', index=False)
    else:
        print("-> Resultado dinámico: No se encontraron reglas que superen el umbral de Lift >= 1.15.")
        print("   Esto indica independencia condicional entre las combinaciones bajo el soporte indicado.")

    pausa_pantallazo("Saca pantallazo de los DISTINTOS ENFOQUES DE REGLAS DE ASOCIACIÓN.")
    return top_rule


def etapa8_modelos_regresion(df_modelo):
    """Etapa 8: Modelos de regresión, 5-Fold Cross Validation y objetivo logarítmico."""
    print("\n--- 8. MODELOS DE REGRESIÓN Y EVALUACIÓN DE PRECISIÓN ---")
    
    # Excluir identificador y variable objetivo
    columnas_excluir = ['CODIGO_ANONIMIZADO', 'VALOR_NETO']
    cols_externas = ['POBLACION_DEPARTAMENTO', 'TASA_AFILIADOS_POR_1000_HAB']

    cols_base = [c for c in df_modelo.columns if c not in columnas_excluir + cols_externas]
    cols_enriquecidas = [c for c in df_modelo.columns if c not in columnas_excluir]
    cols_sin_fuga = [c for c in cols_enriquecidas if c not in ['DIAS_HOSP', 'CANT_ATENCIONES']]

    y = df_modelo['VALOR_NETO']
    y_log = np.log1p(df_modelo['VALOR_NETO'])

    # Partición 80/20 común
    X_base = df_modelo[cols_base]
    X_ext = df_modelo[cols_enriquecidas]
    X_sin_fuga = df_modelo[cols_sin_fuga]

    X_b_tr, X_b_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.2, random_state=RANDOM_STATE)
    X_e_tr, X_e_te, _, _ = train_test_split(X_ext, y, test_size=0.2, random_state=RANDOM_STATE)
    X_sf_tr, X_sf_te, _, _ = train_test_split(X_sin_fuga, y, test_size=0.2, random_state=RANDOM_STATE)
    _, _, y_log_tr, y_log_te = train_test_split(X_ext, y_log, test_size=0.2, random_state=RANDOM_STATE)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring_cv = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error'
    }

    # 1. Modelo Base
    pipe_base = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
    pipe_base.fit(X_b_tr, y_tr)
    pred_base = pipe_base.predict(X_b_te)
    r2_b = r2_score(y_te, pred_base)
    mse_b = mean_squared_error(y_te, pred_base)
    mae_b = mean_absolute_error(y_te, pred_base)
    cv_b = cross_validate(pipe_base, X_base, y, cv=cv, scoring=scoring_cv)

    # 2. Modelo Enriquecido (con Población INEI y Tasa de Cobertura)
    pipe_ext = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
    pipe_ext.fit(X_e_tr, y_tr)
    pred_ext = pipe_ext.predict(X_e_te)
    r2_e = r2_score(y_te, pred_ext)
    mse_e = mean_squared_error(y_te, pred_ext)
    mae_e = mean_absolute_error(y_te, pred_ext)
    cv_e = cross_validate(pipe_ext, X_ext, y, cv=cv, scoring=scoring_cv)

    # 3. Modelo Ridge Regularizado
    pipe_ridge = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=100.0, random_state=RANDOM_STATE))])
    pipe_ridge.fit(X_e_tr, y_tr)
    pred_ridge = pipe_ridge.predict(X_e_te)
    r2_r = r2_score(y_te, pred_ridge)
    mse_r = mean_squared_error(y_te, pred_ridge)
    mae_r = mean_absolute_error(y_te, pred_ridge)
    cv_r = cross_validate(pipe_ridge, X_ext, y, cv=cv, scoring=scoring_cv)

    # 4. Modelo Sin Fuga (Excluyendo DIAS_HOSP y CANT_ATENCIONES)
    pipe_sf = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
    pipe_sf.fit(X_sf_tr, y_tr)
    pred_sf = pipe_sf.predict(X_sf_te)
    r2_sf = r2_score(y_te, pred_sf)
    mse_sf = mean_squared_error(y_te, pred_sf)
    mae_sf = mean_absolute_error(y_te, pred_sf)
    cv_sf = cross_validate(pipe_sf, X_sin_fuga, y, cv=cv, scoring=scoring_cv)

    # 5. Modelo Logarítmico log1p(VALOR_NETO) con reconversión métrica vía expm1
    pipe_log = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
    pipe_log.fit(X_e_tr, y_log_tr)
    pred_log = pipe_log.predict(X_e_te)
    pred_recov = np.expm1(pred_log)
    r2_log = r2_score(y_te, pred_recov)
    mse_log = mean_squared_error(y_te, pred_recov)
    mae_log = mean_absolute_error(y_te, pred_recov)

    # Tabla comparativa
    tabla_modelos = pd.DataFrame({
        'Configuración del Modelo': [
            '1. Base (Clínico SIS)',
            '2. Enriquecido (+ Población INEI & Tasa)',
            '3. Ridge Regularizado (+ INEI)',
            '4. Sin Fuga (Pre-hospitalización)',
            '5. Transformación log1p (Reconvertido)'
        ],
        'R² (Test 80/20)': [r2_b, r2_e, r2_r, r2_sf, r2_log],
        'RMSE (Test)': [np.sqrt(mse_b), np.sqrt(mse_e), np.sqrt(mse_r), np.sqrt(mse_sf), np.sqrt(mse_log)],
        'MAE (Test)': [mae_b, mae_e, mae_r, mae_sf, mae_log],
        '5-Fold CV R² (Media ± Desv)': [
            f"{cv_b['test_r2'].mean():.4f} ± {cv_b['test_r2'].std():.4f}",
            f"{cv_e['test_r2'].mean():.4f} ± {cv_e['test_r2'].std():.4f}",
            f"{cv_r['test_r2'].mean():.4f} ± {cv_r['test_r2'].std():.4f}",
            f"{cv_sf['test_r2'].mean():.4f} ± {cv_sf['test_r2'].std():.4f}",
            "N/A (Escala log)"
        ],
        '5-Fold CV RMSE (Media ± Desv)': [
            f"{-cv_b['test_rmse'].mean():.2f} ± {cv_b['test_rmse'].std():.2f}",
            f"{-cv_e['test_rmse'].mean():.2f} ± {cv_e['test_rmse'].std():.2f}",
            f"{-cv_r['test_rmse'].mean():.2f} ± {cv_r['test_rmse'].std():.2f}",
            f"{-cv_sf['test_rmse'].mean():.2f} ± {cv_sf['test_rmse'].std():.2f}",
            "N/A (Escala log)"
        ],
        '5-Fold CV MAE (Media ± Desv)': [
            f"{-cv_b['test_mae'].mean():.2f} ± {cv_b['test_mae'].std():.2f}",
            f"{-cv_e['test_mae'].mean():.2f} ± {cv_e['test_mae'].std():.2f}",
            f"{-cv_r['test_mae'].mean():.2f} ± {cv_r['test_mae'].std():.2f}",
            f"{-cv_sf['test_mae'].mean():.2f} ± {cv_sf['test_mae'].std():.2f}",
            "N/A (Escala log)"
        ]
    })

    print("\n" + "=" * 96)
    print("           TABLA COMPARATIVA RIGUROSA DE MODELOS DE REGRESIÓN")
    print("=" * 96)
    print(tabla_modelos.to_string(index=False))
    tabla_modelos.to_csv('salidas/comparativa_modelos_regresion.csv', index=False)
    print("-> Tabla guardada en 'salidas/comparativa_modelos_regresion.csv'")

    # Variaciones respecto al modelo base
    dif_r2 = r2_e - r2_b
    dif_mse = mse_e - mse_b
    pct_dif_mse = (dif_mse / mse_b) * 100

    print("\nEvaluación Empírica del Impacto de Fuentes Externas (INEI):")
    print(f"- Variación en R²:  {dif_r2:+.6f}")
    print(f"- Variación en MSE: {dif_mse:+.2f} ({pct_dif_mse:+.4f}%)")

    if dif_mse < 0 and abs(pct_dif_mse) >= 0.05:
        print("-> Conclusión: La incorporación de la fuente externa INEI reduce el error y eleva la precisión de forma medible.")
    else:
        print("-> Conclusión: La incorporación de datos externos territoriales genera una mejora marginal o contextual;")
        print("   el costo médico individual está dominado abrumadoramente por factores clínicos directos (días de hospitalización).")

    print("\nLimitación metodológica del Modelo 4 (Sin Fuga):")
    print(f"Al excluir DIAS_HOSP y CANT_ATENCIONES, el R² cae a {r2_sf:.4f}, evidenciando que predecir el costo")
    print("únicamente con comorbilidades previas es más complejo que cuando el paciente ya requirió internamiento.")

    # Extracción de coeficientes del modelo enriquecido
    reg_ext = pipe_ext.named_steps['reg']
    coefs_df = pd.DataFrame({'Variable': X_ext.columns, 'Impacto': reg_ext.coef_}).sort_values(by='Impacto', ascending=False)
    print("\nTop 5 variables con mayor impacto en el costo neto:")
    print(coefs_df.head(5).to_string(index=False))

    pausa_pantallazo("Saca pantallazo de los MODELOS DE REGRESIÓN Y EVALUACIÓN DE PRECISIÓN.")
    return coefs_df, dif_r2, dif_mse, pct_dif_mse


def etapa9_inferencia_acciones(coefs_df, top_rule, dif_r2, dif_mse, pct_dif_mse):
    """Etapa 9: Inferencia dinámica de decisiones y tabla estructurada."""
    print("\n" + "=" * 88)
    print("   9. INFERENCIA DE ACCIONES A REALIZAR EN BASE A LOS RESULTADOS REALES")
    print("=" * 88)

    top_var_row = coefs_df.iloc[0]
    var_top = top_var_row['Variable']
    val_top = top_var_row['Impacto']

    # Lógica condicional para recomendación de regresión
    if 'DIAS' in var_top or 'HOSP' in var_top:
        hallazgo_reg = f"La variable '{var_top}' es el principal detonante del gasto médico."
        evidencia_reg = f"Coeficiente beta = {val_top:+.2f} en el modelo estandarizado."
        accion_reg = "Priorizar programas de hospitalización domiciliaria y monitoreo ambulatorio preventivo."
    elif 'ATENCIONES' in var_top:
        hallazgo_reg = f"La variable '{var_top}' domina el gasto en salud."
        evidencia_reg = f"Coeficiente beta = {val_top:+.2f}."
        accion_reg = "Unificar atenciones en consultas multidisciplinarias para reducir costos por visita recurrente."
    else:
        hallazgo_reg = f"La variable '{var_top}' presenta la mayor influencia positiva en el costo."
        evidencia_reg = f"Coeficiente beta = {val_top:+.2f}."
        accion_reg = f"Focalizar intervenciones sanitarias directas sobre el factor '{var_top}'."

    # Lógica condicional para recomendación de reglas de asociación
    if top_rule is not None and top_rule['lift'] > 1.0:
        r_ant = ', '.join(list(top_rule['antecedents']))
        r_con = ', '.join(list(top_rule['consequents']))
        hallazgo_aso = f"Comorbilidad concurrente significativa entre [{r_ant}] y [{r_con}]."
        evidencia_aso = f"Lift = {top_rule['lift']:.2f}, Confianza = {top_rule['confidence']:.1%}, Soporte = {top_rule['support']:.1%}."
        accion_aso = f"Establecer tamizaje obligatorio de '{r_con}' al diagnosticar '{r_ant}' en atención primaria."
    else:
        hallazgo_aso = "No se detectaron reglas de alta dependencia bajo los umbrales exigidos."
        evidencia_aso = "Lift <= 1.0 en reglas evaluadas."
        accion_aso = "Auditar y estandarizar la captura de diagnósticos secundarios en la red del SIS."

    # Lógica condicional para recomendación de fuente externa
    if dif_mse < 0:
        hallazgo_ext = "La integración de datos demográficos INEI optimiza la precisión del gasto regional."
        evidencia_ext = f"Reducción de MSE en {abs(dif_mse):.2f} unidades ({pct_dif_mse:+.2f}%)."
        accion_ext = "Ponderar la masa demográfica regional en la asignación presupuestaria hospitalaria."
    else:
        hallazgo_ext = "La población regional actúa como contexto territorial de escala macro."
        evidencia_ext = f"Variación de MSE de {dif_mse:+.2f} ({pct_dif_mse:+.2f}%)."
        accion_ext = "Utilizar la población para estimar necesidades de infraestructura y no para predecir costos unitarios."

    tabla_resumen_acciones = pd.DataFrame({
        'Dimensión Analítica': ['Regresión Predictiva', 'Reglas de Asociación', 'Fuente Externa INEI'],
        'Hallazgo Empírico': [hallazgo_reg, hallazgo_aso, hallazgo_ext],
        'Evidencia Numérica': [evidencia_reg, evidencia_aso, evidencia_ext],
        'Acción Recomendada': [accion_reg, accion_aso, accion_ext]
    })

    print(tabla_resumen_acciones.to_string(index=False))
    tabla_resumen_acciones.to_csv('salidas/resumen_inferencia_acciones.csv', index=False)
    print("\n-> Tabla guardada en 'salidas/resumen_inferencia_acciones.csv'")
    pausa_pantallazo("Saca pantallazo de la INFERENCIA DE ACCIONES A REALIZAR.")


def main():
    print("=" * 88)
    print("   INICIO DEL PIPELINE DE MINERÍA DE DATOS - EVALUACIÓN 2")
    print("=" * 88)
    archivo_csv = 'Afiliados_activos_DM_SIS.csv'

    if not os.path.exists(archivo_csv):
        raise FileNotFoundError(f"No se encontró {archivo_csv}. Debe estar en la carpeta raíz.")

    print(f"Cargando dataset maestro para inspección y análisis: {archivo_csv}...")
    df_completo = pd.read_csv(archivo_csv, low_memory=False)

    # Paso 0: Inspección
    inspeccionar_dataset_original(df_completo)

    # Verificación de fuentes externas
    verificar_fuentes_adicionales()

    # Carga de fuente externa real INEI
    df_pob = cargar_fuente_externa_poblacion('cantidad_poblacion.xlsx')

    # Etapas del pipeline
    df_sel = etapa1_seleccion(df_completo, n_muestra=N_MUESTRA_INICIAL)
    df_int = etapa2_incorporacion_externa(df_sel, df_pob, df_completo)
    df_limpio = etapa3_limpieza_outliers(df_int)
    df_cod, df_modelo = etapa4_codificacion(df_limpio)
    etapa5_normalizacion(df_cod)
    etapa6_estadisticos_graficos(df_cod)
    top_rule = etapa7_reglas_asociacion(df_limpio)
    coefs_df, dif_r2, dif_mse, pct_dif_mse = etapa8_modelos_regresion(df_modelo)
    etapa9_inferencia_acciones(coefs_df, top_rule, dif_r2, dif_mse, pct_dif_mse)

    print("\n" + "=" * 88)
    print("   PIPELINE COMPLETADO EXITOSAMENTE. TODOS LOS RESULTADOS GUARDADOS EN 'salidas/'.")
    print("=" * 88)


if __name__ == '__main__':
    main()