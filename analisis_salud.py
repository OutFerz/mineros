import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Selección de datos
print("--- 1. CARGA Y SELECCIÓN DE DATOS ---")
archivo_csv = 'Afiliados_activos_DM_SIS.csv'

# Cargar el dataset
try:
    df = pd.read_csv(archivo_csv)
    print(f"Dataset cargado exitosamente. Total de registros: {df.shape[0]}")
    print(df.head())
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {archivo_csv} en la carpeta raíz.")

# Aquí puedes continuar con el resto de los pasos...
# Para mostrar un gráfico y sacarle el pantallazo, usarás plt.show():
# plt.figure(figsize=(10,6))
# sns.histplot(df['alguna_columna'])
# plt.show()  <-- Esto abrirá una ventana en tu PC con el gráfico