import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules

def main():
    print("="*50)
    print("PROYECTO DE MINERÍA DE DATOS - ANÁLISIS DE SALUD")
    print("="*50)

    # 1. CREACIÓN DE DATASET (Si no existe, se crea uno sintético de 500.000 registros)
    archivo_csv = 'dataset_salud_kaggle.csv'
    archivo_secundario = 'datos_demograficos_extra.csv'

    if not os.path.exists(archivo_csv):
        print(f"\n[!] No se encontró {archivo_csv}. Generando dataset sintético de 500,000 registros para demostración...")
        np.random.seed(42)
        n_registros = 500000
        
        datos = {
            'ID_Paciente': range(1, n_registros + 1),
            'Edad': np.random.randint(18, 90, n_registros),
            'Sexo': np.random.choice(['M', 'F'], n_registros),
            'Region': np.random.choice(['Norte', 'Centro', 'Sur', 'Metropolitana'], n_registros),
            'Diagnostico_Principal': np.random.choice(['Diabetes', 'Hipertension', 'Asma', 'Ninguno', 'Cardiopatia'], n_registros, p=[0.15, 0.25, 0.1, 0.4, 0.1]),
            'Tratamiento_Semanas': np.random.randint(1, 52, n_registros),
            'Costo_Tratamiento': np.random.uniform(50, 5000, n_registros),
            'IMC': np.random.normal(26, 4, n_registros)
        }
        
        # Introducir algunos nulos para la limpieza
        df_base = pd.DataFrame(datos)
        # Fix deprecated code using modern pandas assignment
        indices_nan = df_base.sample(frac=0.05).index
        df_base.loc[indices_nan, 'IMC'] = np.nan
        df_base.to_csv(archivo_csv, index=False)
        print(f"[OK] {archivo_csv} generado con éxito.")
    
    if not os.path.exists(archivo_secundario):
        print(f"\n[!] Generando fuente de datos extra: {archivo_secundario}...")
        df_extra = pd.DataFrame({
            'Region': ['Norte', 'Centro', 'Sur', 'Metropolitana'],
            'Indice_Pobreza_Regional': [15.2, 12.1, 14.5, 9.8],
            'Acceso_Hospitales': ['Bajo', 'Medio', 'Bajo', 'Alto']
        })
        df_extra.to_csv(archivo_secundario, index=False)
        print(f"[OK] {archivo_secundario} generado con éxito.")

    # 2. INCORPORACIÓN DE DATOS DE OTRAS FUENTES
    print("\n--- 1. CARGA E INCORPORACIÓN DE FUENTES ---")
    df_principal = pd.read_csv(archivo_csv)
    df_secundario = pd.read_csv(archivo_secundario)
    
    # Merge de ambas fuentes
    df = pd.merge(df_principal, df_secundario, on='Region', how='left')
    print(f"Dataset combinado cargado. Total de registros: {df.shape[0]}, Columnas: {df.shape[1]}")
    
    # 3. SELECCIÓN DE DATOS
    print("\n--- 2. SELECCIÓN DE DATOS ---")
    # Seleccionamos las columnas más relevantes para nuestro análisis
    columnas_seleccionadas = ['Edad', 'Sexo', 'Diagnostico_Principal', 'Tratamiento_Semanas', 'Costo_Tratamiento', 'IMC', 'Indice_Pobreza_Regional']
    df_seleccion = df[columnas_seleccionadas].copy()
    print("Columnas seleccionadas:", columnas_seleccionadas)

    # 4. LIMPIEZA DE DATOS
    print("\n--- 3. LIMPIEZA DE DATOS ---")
    nulos_antes = df_seleccion.isnull().sum().sum()
    print(f"Valores nulos antes de limpieza: {nulos_antes}")
    # Imputar IMC con la mediana
    mediana_imc = df_seleccion['IMC'].median()
    df_seleccion['IMC'] = df_seleccion['IMC'].fillna(mediana_imc)
    nulos_despues = df_seleccion.isnull().sum().sum()
    print(f"Valores nulos después de limpieza: {nulos_despues}")

    # 5. CODIFICACIÓN DE DATOS
    print("\n--- 4. CODIFICACIÓN DE DATOS ---")
    # Codificar variables categóricas (Sexo, Diagnostico_Principal)
    le_sexo = LabelEncoder()
    le_diag = LabelEncoder()
    df_seleccion['Sexo_Cod'] = le_sexo.fit_transform(df_seleccion['Sexo'])
    df_seleccion['Diagnostico_Cod'] = le_diag.fit_transform(df_seleccion['Diagnostico_Principal'])
    print("Muestra de datos codificados:")
    print(df_seleccion[['Sexo', 'Sexo_Cod', 'Diagnostico_Principal', 'Diagnostico_Cod']].head())

    # 6. NORMALIZACIÓN DE DATOS
    print("\n--- 5. NORMALIZACIÓN DE DATOS ---")
    scaler = MinMaxScaler()
    df_seleccion[['Edad_Norm', 'Costo_Norm', 'IMC_Norm']] = scaler.fit_transform(df_seleccion[['Edad', 'Costo_Tratamiento', 'IMC']])
    print("Muestra de datos normalizados:")
    print(df_seleccion[['Edad', 'Edad_Norm', 'Costo_Tratamiento', 'Costo_Norm']].head())

    # 7. APLICAR PRINCIPALES ESTADÍSTICOS
    print("\n--- 6. ANÁLISIS ESTADÍSTICO ---")
    print("Estadísticos descriptivos:")
    print(df_seleccion[['Edad', 'Tratamiento_Semanas', 'Costo_Tratamiento', 'IMC', 'Indice_Pobreza_Regional']].describe())
    
    # Visualización estadística
    plt.figure(figsize=(10,6))
    sns.heatmap(df_seleccion[['Edad', 'Tratamiento_Semanas', 'Costo_Tratamiento', 'IMC', 'Sexo_Cod', 'Diagnostico_Cod']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación")
    print("\n[!] Cierra la ventana del gráfico para continuar...")
    plt.show()

    # 8. REGLAS DE ASOCIACIÓN (Apriori)
    print("\n--- 7. REGLAS DE ASOCIACIÓN ---")
    # Discretizamos para crear transacciones (Ej: Edad Mayor a 50, IMC Alto, Costo Alto)
    df_asociacion = pd.DataFrame()
    df_asociacion['Mayor_50_Anios'] = df_seleccion['Edad'] > 50
    df_asociacion['IMC_Sobrepeso'] = df_seleccion['IMC'] > 25
    df_asociacion['Costo_Elevado'] = df_seleccion['Costo_Tratamiento'] > 2500
    df_asociacion['Es_Diabetico'] = df_seleccion['Diagnostico_Principal'] == 'Diabetes'
    
    # Aplicar Apriori en una muestra para rendimiento (o en todo si hay memoria)
    muestra_aso = df_asociacion.sample(50000, random_state=1)
    frequent_itemsets = apriori(muestra_aso, min_support=0.05, use_colnames=True)
    if not frequent_itemsets.empty:
        reglas = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        print("Reglas de Asociación encontradas (Top 5 por Lift):")
        print(reglas.sort_values('lift', ascending=False).head()[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        print("No se encontraron itemsets frecuentes con el soporte mínimo indicado.")

    # 9. MODELOS DE REGRESIÓN
    print("\n--- 8. MODELO DE REGRESIÓN MULTIPLE ---")
    # Predecir el Costo_Tratamiento basado en Edad, IMC, Tratamiento_Semanas e Indice_Pobreza_Regional
    X = df_seleccion[['Edad_Norm', 'IMC_Norm', 'Tratamiento_Semanas', 'Indice_Pobreza_Regional']]
    y = df_seleccion['Costo_Tratamiento']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_reg = LinearRegression()
    modelo_reg.fit(X_train, y_train)
    y_pred = modelo_reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinación (R2): {r2:.4f}")
    
    plt.figure(figsize=(8,5))
    plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.3) # Scatter de 1000 puntos para visibilidad
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Costo Real')
    plt.ylabel('Costo Predicho')
    plt.title('Regresión Lineal: Costo Real vs Predicho')
    print("\n[!] Cierra la ventana del gráfico para continuar...")
    plt.show()

    # 10. INFERIR ACCIONES
    print("\n--- 9. INFERENCIA DE ACCIONES (CONCLUSIONES) ---")
    print("En base a los resultados obtenidos, podemos inferir:")
    print("1. Optimización de Recursos: Si se identifica una fuerte regla de asociación entre 'Mayor de 50' e 'IMC Sobrepeso' con diagnósticos cardíacos, se deben realizar campañas preventivas dirigidas a este grupo.")
    print("2. Presupuesto Hospitalario: El modelo de regresión permite estimar el costo de tratamiento de nuevos pacientes según su perfil, ayudando en la planificación de presupuesto.")
    print("3. Políticas Regionales: Al incorporar la variable 'Indice_Pobreza_Regional', observamos el impacto del contexto demográfico, sugiriendo redirigir fondos a regiones de alto índice para mejorar el acceso a salud.")
    print("\nEjecución finalizada con éxito.")

if __name__ == '__main__':
    main()
