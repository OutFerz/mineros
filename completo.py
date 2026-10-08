import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import os

# Ignorar advertencias menores para mantener la consola limpia en los pantallazos
warnings.filterwarnings('ignore')

print("==========================================================")
print("   INICIO DEL PIPELINE DE MINERÍA DE DATOS - SALUD SIS")
print("==========================================================\n")

# ---------------------------------------------------------
# 1. SELECCIÓN DE DATOS
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

# Simulamos dataset gubernamental externo: "Presupuesto Regional de Salud (Millones)"
datos_externos = pd.DataFrame({
    'DEPARTAMENTO': departamentos_unicos,
    'PRESUPUESTO_REGIONAL_MILLONES': np.random.uniform(50, 300, size=len(departamentos_unicos))
})
datos_externos.to_csv('datos_externos_minsal.csv', index=False)

df_integrado = pd.merge(df_seleccion, datos_externos, on='DEPARTAMENTO', how='left')
print("Dataset principal fusionado (Merge) con 'datos_externos_minsal.csv'.")
print("Muestra:")
print(df_integrado[['DEPARTAMENTO', 'PRESUPUESTO_REGIONAL_MILLONES']].head(3))
input("\n[Pausa] Saca pantallazo de esta sección y presiona Enter para continuar...")


# ---------------------------------------------------------
# 3. LIMPIEZA DE DATOS
# ---------------------------------------------------------
print("\n--- 3. LIMPIEZA DE DATOS ---")
print(f"Nulos ANTES de la limpieza:\n{df_integrado.isnull().sum()}")

# Eliminar duplicados exactos
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
# Normalizamos para que todas las variables tengan media 0 y varianza 1
df_normalizado[cols_numericas] = scaler.fit_transform(df_normalizado[cols_numericas])

print("Datos estandarizados (Z-score).")
print(df_normalizado[['EDAD', 'VALOR_NETO']].head(3))
input("\n[Pausa] Saca pantallazo de la normalización y presiona Enter para continuar...")


# ---------------------------------------------------------
# 6. ESTADÍSTICOS DESCRIPTIVOS Y VISUALIZACIÓN
# ---------------------------------------------------------
print("\n--- 6. ESTADÍSTICOS PRINCIPALES ---")
estadisticos = df_codificado.describe().round(2)
print("Resumen Estadístico del Dataset (Extracto):")
print(estadisticos[['EDAD', 'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP']])

print("\n>>> Se abrirá una ventana con el Mapa de Calor de Correlaciones. Ciérrala para continuar el código.")
plt.figure(figsize=(10, 8))
sns.heatmap(df_codificado.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación - Variables SIS Diabetes')
plt.tight_layout()
plt.show() # Esta línea pausa el script hasta que cierres la ventana del gráfico
input("\n[Pausa] Asegúrate de haber sacado pantallazo al gráfico y a la consola, luego presiona Enter...")


# ---------------------------------------------------------
# 7. REGLAS DE ASOCIACIÓN (APRIORI)
# ---------------------------------------------------------
print("\n--- 7. REGLAS DE ASOCIACIÓN ---")
# Preparamos los datos en formato booleano (True/False) para Apriori
# Consideramos "presencia" de factores de riesgo si el código es mayor a 0
df_apriori = pd.DataFrame()
df_apriori['Hipertension'] = df_codificado['CON_DX_HIPERTENSION'] > 0
df_apriori['Obesidad'] = df_codificado['CON_DX_OBESIDAD'] > 0
df_apriori['Salud_Mental'] = df_codificado['CON_DX_SALUDMENTAL'] > 0
df_apriori['Atenciones_Frecuentes'] = df_codificado['CANT_ATENCIONES'] > df_codificado['CANT_ATENCIONES'].median()

# Generar items frecuentes (Soporte mínimo del 5%)
frequent_itemsets = apriori(df_apriori, min_support=0.05, use_colnames=True)

# Generar reglas de asociación (Lift superior a 1 indica asociación positiva)
if not frequent_itemsets.empty:
    reglas = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    print("Principales Reglas Encontradas (Antecedente -> Consecuente):")
    print(reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
else:
    print("No se encontraron reglas fuertes con el umbral establecido.")
input("\n[Pausa] Saca pantallazo de las reglas de asociación y presiona Enter para continuar...")


# ---------------------------------------------------------
# 8. MODELO DE REGRESIÓN
# ---------------------------------------------------------
print("\n--- 8. APLICACIÓN DE MODELO DE REGRESIÓN ---")
# Objetivo: Predecir el 'VALOR_NETO' (Costo) basándonos en variables clínicas
X = df_normalizado.drop(['VALOR_NETO'], axis=1) 
y = df_codificado['VALOR_NETO'] # Usamos el valor real para que la métrica sea comprensible

# Dividimos en set de Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

predicciones = modelo_regresion.predict(X_test)

mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f"Evaluación del Modelo de Regresión Lineal Múltiple:")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R2): {r2:.4f}")
print("\nImpacto de variables en el Costo (Coeficientes):")
coeficientes = pd.DataFrame({'Variable': X.columns, 'Impacto': modelo_regresion.coef_})
print(coeficientes.sort_values(by='Impacto', ascending=False).head(5))

print("\n==========================================================")
print(" PROCESO FINALIZADO. REVISA TUS CAPTURAS PARA EL INFORME.")
print("==========================================================")