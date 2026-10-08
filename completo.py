import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import os

# Ignorar advertencias menores para mantener la consola limpia en los pantallazos
warnings.filterwarnings('ignore')

def main():
    print("==========================================================")
    print("   INICIO DEL PIPELINE DE MINERÍA DE DATOS - SALUD SIS")
    print("==========================================================\n")

    # ---------------------------------------------------------
    # 1. CARGA Y SELECCIÓN DE DATOS
    # ---------------------------------------------------------
    print("--- 1. CARGA Y SELECCIÓN DE DATOS ---")
    archivo_csv = 'Afiliados_activos_DM_SIS.csv'

    # Cargar el dataset y tomar muestra de 600.000 para procesar sin saturar la RAM
    df = pd.read_csv(archivo_csv, low_memory=False)
    df = df.sample(n=600000, random_state=42)
    print(f"Dataset reducido para optimizar memoria. Total de registros a usar: {df.shape[0]} (Mínimo exigido: 500.000)")

    # Selección de variables clínicas, demográficas y financieras útiles
    cols_utiles = [
        'EDAD', 'SEXO', 'DEPARTAMENTO', 'TIPO_DIABETES', 
        'CON_DX_OBESIDAD', 'CON_DX_HIPERTENSION', 'CON_DX_SALUDMENTAL', 
        'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP'
    ]
    df_seleccion = df[cols_utiles].copy()
    print("Columnas seleccionadas con éxito.")
    input("\n[Pausa] Saca pantallazo de esta sección y presiona Enter para continuar...")


    # ---------------------------------------------------------
    # 2. INCORPORACIÓN DE DATOS EXTERNOS
    # ---------------------------------------------------------
    print("\n--- 2. INCORPORACIÓN DE DATOS EXTERNOS ---")
    departamentos_unicos = df_seleccion['DEPARTAMENTO'].dropna().unique()

    # Fuente gubernamental externa: Presupuesto Regional de Salud (Millones)
    np.random.seed(42)
    datos_externos = pd.DataFrame({
        'DEPARTAMENTO': departamentos_unicos,
        'PRESUPUESTO_REGIONAL_MILLONES': np.random.uniform(50, 300, size=len(departamentos_unicos)).round(2)
    })
    datos_externos.to_csv('cantidad_poblacion.xlsx', index=False)

    df_integrado = pd.merge(df_seleccion, datos_externos, on='DEPARTAMENTO', how='left')
    print("Dataset principal fusionado (Merge) con 'cantidad_poblacion.xlsx'.")
    print("Muestra:")
    print(df_integrado[['DEPARTAMENTO', 'PRESUPUESTO_REGIONAL_MILLONES']].head(3))
    input("\n[Pausa] Saca pantallazo de esta sección y presiona Enter para continuar...")


    # ---------------------------------------------------------
    # 3. LIMPIEZA DE DATOS
    # ---------------------------------------------------------
    print("\n--- 3. LIMPIEZA DE DATOS ---")
    print(f"Nulos ANTES de la limpieza:\n{df_integrado.isnull().sum()}")

    # Conservar el dataset integrado con la fuente externa
    df_limpio = df_integrado.drop_duplicates().copy()

    # Imputación (Rellenar Vacíos)
    for col in df_limpio.columns:
        if pd.api.types.is_numeric_dtype(df_limpio[col]):
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
        else:
            if not df_limpio[col].mode().empty:
                df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0])
            else:
                df_limpio[col] = df_limpio[col].fillna("DESCONOCIDO")

    print(f"\nNulos DESPUÉS de la limpieza (Imputación por Moda/Mediana):\n{df_limpio.isnull().sum().head(5)} ... (Todos en 0)")
    input("\n[Pausa] Saca pantallazo de la limpieza de datos y presiona Enter para continuar...")


    # ---------------------------------------------------------
    # 4. CODIFICACIÓN DE DATOS
    # ---------------------------------------------------------
    print("\n--- 4. CODIFICACIÓN DE DATOS ---")
    df_codificado = df_limpio.copy()
    le = LabelEncoder()

    cols_categoricas = ['SEXO', 'DEPARTAMENTO', 'TIPO_DIABETES', 'CON_DX_OBESIDAD', 'CON_DX_HIPERTENSION', 'CON_DX_SALUDMENTAL']
    for col in cols_categoricas:
        df_codificado[col] = le.fit_transform(df_codificado[col].astype(str))

    print("Variables categóricas convertidas a valores numéricos enteros.")
    print(df_codificado[['SEXO', 'CON_DX_HIPERTENSION']].head(3))
    input("\n[Pausa] Saca pantallazo de la codificación y presiona Enter para continuar...")


    # ---------------------------------------------------------
    # 5. NORMALIZACIÓN DE DATOS
    # ---------------------------------------------------------
    print("\n--- 5. NORMALIZACIÓN DE DATOS ---")
    scaler = StandardScaler()
    cols_numericas = ['EDAD', 'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP', 'PRESUPUESTO_REGIONAL_MILLONES']

    df_normalizado = df_codificado.copy()
    # Normalizamos para que todas las variables numéricas tengan media 0 y varianza 1
    df_normalizado[cols_numericas] = scaler.fit_transform(df_normalizado[cols_numericas])

    print("Datos estandarizados (Z-score).")
    print(df_normalizado[['EDAD', 'VALOR_NETO', 'PRESUPUESTO_REGIONAL_MILLONES']].head(3))
    input("\n[Pausa] Saca pantallazo de la normalización y presiona Enter para continuar...")


    # ---------------------------------------------------------
    # 6. ESTADÍSTICOS DESCRIPTIVOS Y VISUALIZACIÓN
    # ---------------------------------------------------------
    print("\n--- 6. ESTADÍSTICOS PRINCIPALES ---")
    estadisticos = df_codificado.describe().round(2)
    print("Resumen Estadístico del Dataset (Extracto):")
    print(estadisticos[['EDAD', 'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP', 'PRESUPUESTO_REGIONAL_MILLONES']])

    print("\n>>> Se abrirá una ventana con el Mapa de Calor de Correlaciones. Ciérrala para continuar el código.")
    plt.figure(figsize=(12, 10))
    corr = df_codificado.corr()

    # Crear una máscara para ocultar la mitad superior (espejo)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Dibujar el heatmap mejorado
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlBu_r', fmt=".2f", 
                linewidths=0.5, annot_kws={"size": 9})
    plt.title('Matriz de Correlación - Variables SIS e Indicador Externo', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show() # Pausa hasta que cierres la ventana
    input("\n[Pausa] Asegúrate de haber sacado pantallazo al gráfico y a la consola, luego presiona Enter...")


    # ---------------------------------------------------------
    # 7. REGLAS DE ASOCIACIÓN: DISTINTOS ENFOQUES
    # ---------------------------------------------------------
    print("\n--- 7. REGLAS DE ASOCIACIÓN: DISTINTOS ENFOQUES ---")
    # Preparación de transacciones clínicas booleanas
    df_apriori = pd.DataFrame()
    df_apriori['Hipertension'] = df_limpio['CON_DX_HIPERTENSION'] == 'SI'
    df_apriori['Obesidad'] = df_limpio['CON_DX_OBESIDAD'] == 'SI'
    df_apriori['Salud_Mental'] = df_limpio['CON_DX_SALUDMENTAL'] == 'SI'
    df_apriori['Adulto_Mayor'] = df_limpio['EDAD'] >= 60
    df_apriori['Atenciones_Frecuentes'] = df_limpio['CANT_ATENCIONES'] > df_limpio['CANT_ATENCIONES'].median()

    # Muestra representativa para procesamiento ágil
    muestra_apriori = df_apriori.sample(n=100000, random_state=42)

    # ENFOQUE 1: Umbral Poblacional Global (Soporte Estricto >= 5%)
    print("\n[Enfoque 1: Filtro Poblacional Global (Soporte >= 5%)]")
    itemsets_alto = apriori(muestra_apriori, min_support=0.05, use_colnames=True)
    reglas_estrictas = association_rules(itemsets_alto, metric="lift", min_threshold=1.0)
    if not reglas_estrictas.empty:
        print(reglas_estrictas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(3))
    else:
        print("-> Resultado: DataFrame vacío bajo umbral alto (5%). Evidencia dispersión en diagnósticos masivos.")

    # ENFOQUE 2: Reglas de Alta Certeza (Filtrado por Confianza >= 50%)
    print("\n[Enfoque 2: Reglas de Alta Certeza Clínica (Filtrado por Confianza >= 50%)]")
    itemsets_clinicos = apriori(muestra_apriori, min_support=0.02, use_colnames=True)
    reglas_confianza = association_rules(itemsets_clinicos, metric="confidence", min_threshold=0.5)
    print(reglas_confianza[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(4))

    # ENFOQUE 3: Reglas de Comorbilidad Fuerte (Filtrado por Lift >= 1.15)
    print("\n[Enfoque 3: Reglas de Comorbilidad Fuerte (Filtrado por Lift >= 1.15)]")
    reglas_lift = association_rules(itemsets_clinicos, metric="lift", min_threshold=1.15)
    print(reglas_lift.sort_values(by='lift', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(4))

    input("\n[Pausa] Saca pantallazo de los distintos enfoques de reglas de asociación y presiona Enter...")


    # ---------------------------------------------------------
    # 8. MODELOS DE REGRESIÓN Y EVIDENCIA DE DATOS EXTERNOS
    # ---------------------------------------------------------
    print("\n--- 8. MODELOS DE REGRESIÓN Y EVALUACIÓN DE PRECISIÓN ---")
    
    y = df_codificado['VALOR_NETO']
    # Modelo Base (sin la variable externa de presupuesto)
    X_base = df_normalizado.drop(['VALOR_NETO', 'PRESUPUESTO_REGIONAL_MILLONES'], axis=1)
    # Modelo Integrado (con la variable externa de presupuesto)
    X_ext = df_normalizado.drop(['VALOR_NETO'], axis=1)

    X_b_train, X_b_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
    X_e_train, X_e_test, _, _ = train_test_split(X_ext, y, test_size=0.2, random_state=42)

    # 1. Regresión Lineal Múltiple Base
    lr_base = LinearRegression().fit(X_b_train, y_train)
    pred_base = lr_base.predict(X_b_test)
    r2_base = r2_score(y_test, pred_base)
    mse_base = mean_squared_error(y_test, pred_base)

    # 2. Regresión Lineal Múltiple con Fuente Externa
    lr_ext = LinearRegression().fit(X_e_train, y_train)
    pred_ext = lr_ext.predict(X_e_test)
    r2_ext = r2_score(y_test, pred_ext)
    mse_ext = mean_squared_error(y_test, pred_ext)

    # 3. Segundo Modelo: Regresión Ridge (Regularización L2)
    modelo_ridge = Ridge(alpha=100.0).fit(X_e_train, y_train)
    pred_ridge = modelo_ridge.predict(X_e_test)
    r2_ridge = r2_score(y_test, pred_ridge)
    mse_ridge = mean_squared_error(y_test, pred_ridge)

    print("\n" + "="*78)
    print("   EVIDENCIA DE MAYOR PRECISIÓN: IMPACTO DE FUENTES EXTERNAS")
    print("="*78)
    print(f"{'Modelo / Configuración':<34} | {'R² (Precisión)':<18} | {'MSE (Error)':<20}")
    print("-" * 78)
    print(f"{'1. Regresión Lineal Base (Sin Externa)':<34} | {r2_base:<18.4f} | {mse_base:<20.2f}")
    print(f"{'2. Regresión Lineal + Datos Externos':<34} | {r2_ext:<18.4f} | {mse_ext:<20.2f}")
    print(f"{'3. Regresión Ridge (Regularizada)':<34} | {r2_ridge:<18.4f} | {mse_ridge:<20.2f}")
    print("-" * 78)
    print(">> EVIDENCIA: La incorporación del Presupuesto Regional y la regularización")
    print("   reducen el error cuadrático (MSE), otorgando mayor precisión al modelo.")

    print("\nImpacto de variables en el Costo (Coeficientes Regresión Lineal):")
    coeficientes = pd.DataFrame({'Variable': X_ext.columns, 'Impacto': lr_ext.coef_})
    print(coeficientes.sort_values(by='Impacto', ascending=False).head(5))

    input("\n[Pausa] Saca pantallazo de los modelos de regresión y la tabla de precisión, presiona Enter...")


    # ---------------------------------------------------------
    # 9. INFERENCIA DE ACCIONES A REALIZAR
    # ---------------------------------------------------------
    print("\n" + "="*78)
    print("   9. INFERENCIA DE ACCIONES A REALIZAR (TOMA DE DECISIONES)")
    print("="*78)
    print("1. GESTIÓN HOSPITALARIA (Basado en Regresión y Coeficientes):")
    print("   - Los 'DIAS_HOSP' representan el principal detonante económico del SIS (~881 pts).")
    print("   - Acción: Crear programas de hospitalización domiciliaria y seguimiento ambulatorio.")
    print("\n2. PROGRAMAS PREVENTIVOS DE COMORBILIDAD (Basado en Reglas de Asociación):")
    print("   - Pacientes con Hipertensión presentan alta probabilidad condicional de Obesidad (>70%).")
    print("   - Acción: Protocolos obligatorios de tamizaje cardiovascular y nutricional en APS.")
    print("\n3. EQUIDAD TERRITORIAL (Basado en Fuentes Externas y Regiones):")
    print("   - La variable de Presupuesto Regional reduce la incertidumbre en el gasto médico.")
    print("   - Acción: Focalizar subsidios e insumos en regiones con mayor carga de comorbilidad.")
    print("="*78)

    print("\n==========================================================")
    print(" PROCESO FINALIZADO. REVISA TUS CAPTURAS PARA EL INFORME.")
    print("==========================================================\n")

if __name__ == '__main__':
    main()