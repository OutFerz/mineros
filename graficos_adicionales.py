"""Gráficos complementarios; no modifica los datos ni los resultados originales.

Ejecutar desde mineros: env/Scripts/python.exe graficos_adicionales.py
"""
from pathlib import Path
import ast
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SALIDAS = Path(__file__).resolve().parent / 'salidas'
MOSTRAR_GRAFICOS = False


def guardar(fig, nombre):
    SALIDAS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(SALIDAS / nombre, dpi=180, bbox_inches='tight')
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close(fig)


def graficar_regresion(tabla, reales=None, predicciones=None, coeficientes=None):
    nombres = ['Base', 'Enriquecido', 'Ridge', 'Sin fuga', 'Log reconvertido']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, columna, titulo in zip(axes, tabla.columns[1:4],
                                   ['R² en prueba (mayor es mejor)', 'RMSE en soles (menor es mejor)', 'MAE en soles (menor es mejor)']):
        valores = tabla[columna].to_numpy()
        ax.barh(nombres, valores, color=['#2471a3', '#148f77', '#7d3c98', '#d68910', '#c0392b'])
        ax.axvline(0, color='gray', linewidth=.8)
        for i, valor in enumerate(valores):
            ax.text(valor, i, f' {valor:.3f}', va='center', fontsize=9)
        ax.set_title(titulo, fontsize=11)
        ax.invert_yaxis()
        ax.margins(x=.25)
    guardar(fig, 'regresion_comparativa_metricas.png')

    fig, ax = plt.subplots(figsize=(9, 5))
    medias, desv = [], []
    for valor in tabla.iloc[:4, 4]:
        media, desviacion = valor.split('±')
        medias.append(float(media)); desv.append(float(desviacion))
    ax.errorbar(nombres[:4], medias, yerr=desv, fmt='o', capsize=6, color='#2471a3')
    ax.set(ylabel='R² (media ± desviación estándar)', title='Estabilidad de regresión: validación cruzada de 5 pliegues')
    ax.grid(axis='y', alpha=.25)
    guardar(fig, 'regresion_validacion_cruzada.png')

    if reales is not None and predicciones is not None:
        y = np.asarray(reales)
        p = np.asarray(predicciones)
        indices = np.random.default_rng(42).choice(len(y), min(5000, len(y)), replace=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y[indices], p[indices], s=8, alpha=.2)
        limites = [min(y.min(), p.min()), max(y.max(), p.max())]
        axes[0].plot(limites, limites, '--', color='red')
        axes[0].set(xlabel='Costo real (soles)', ylabel='Costo predicho (soles)', title='Enriquecido: real frente a predicho (hasta 5.000 casos)')
        axes[1].scatter(p[indices], (y-p)[indices], s=8, alpha=.2)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set(xlabel='Costo predicho (soles)', ylabel='Residuo real − predicho (soles)', title='Residuos en la misma muestra de prueba')
        guardar(fig, 'regresion_predicciones_residuos.png')
    if coeficientes is not None:
        seleccion = coeficientes.loc[coeficientes['Impacto'].abs().nlargest(10).index].sort_values('Impacto')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(seleccion['Variable'], seleccion['Impacto'], color='#148f77')
        ax.set(xlabel='Coeficiente por una desviación estándar del predictor (soles)', title='Diez coeficientes de mayor magnitud: modelo enriquecido')
        guardar(fig, 'regresion_coeficientes.png')


def etiqueta(valor):
    if isinstance(valor, str):
        valor = ast.literal_eval(valor.removeprefix('frozenset(').removesuffix(')'))
    return ', '.join(sorted(valor)).replace('_', ' ')


def graficar_asociacion(reglas):
    if reglas.empty:
        return
    top = reglas.head(10).reset_index(drop=True)
    etiquetas = [f'R{i+1}: ' + textwrap.fill(etiqueta(r.antecedents) + ' → ' + etiqueta(r.consequents), 65)
                 for i, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(etiquetas, top['lift'], color='#7d3c98')
    ax.axvline(1, color='red', linestyle='--', label='Independencia: lift = 1')
    ax.invert_yaxis()
    ax.set(xlabel='Lift', title='Diez reglas con mayor lift (incluye reglas inversas)')
    ax.legend()
    guardar(fig, 'asociacion_top_lift.png')
    fig, ax = plt.subplots(figsize=(9, 6))
    puntos = ax.scatter(top['support']*100, top['confidence']*100, c=top['lift'], s=100, cmap='viridis')
    for i, r in top.iterrows():
        ax.annotate(f'R{i+1}', (r['support']*100, r['confidence']*100), xytext=(5, 5), textcoords='offset points')
    fig.colorbar(puntos, ax=ax, label='Lift')
    ax.set(xlabel='Soporte (%)', ylabel='Confianza (%)', title='Frecuencia, confianza y fuerza de asociación')
    guardar(fig, 'asociacion_soporte_confianza.png')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(1, len(top)+1)
    axes[0].plot(x, top['lift'], 'o-', color='#7d3c98')
    axes[0].set(ylabel='Lift', title='Sucesión ordenada de reglas por lift (no es una serie temporal)')
    axes[1].plot(x, top['confidence']*100, 'o-', label='Confianza (%)')
    axes[1].plot(x, top['support']*100, 's-', label='Soporte (%)')
    axes[1].set(xlabel='Posición en el ranking: R1 a R10', ylabel='Porcentaje', xticks=x)
    axes[1].legend()
    guardar(fig, 'sucesion_reglas_metricas.png')


def main():
    graficar_regresion(pd.read_csv(SALIDAS / 'comparativa_modelos_regresion.csv'))
    graficar_asociacion(pd.read_csv(SALIDAS / 'reglas_asociacion_top_lift.csv'))
    print('Cinco gráficos complementarios generados desde los CSV existentes.')


if __name__ == '__main__':
    main()
