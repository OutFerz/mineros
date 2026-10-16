"""Genera una síntesis fiel a las métricas exportadas, sin reentrenar modelos."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent


def main():
    tabla = pd.read_csv(BASE / 'salidas/comparativa_modelos_regresion.csv')
    valores = tabla.iloc[[1, 0, 3], 1].to_numpy() * 100
    nombres = [
        'Modelo enriquecido\nBase + población INEI y tasa regional',
        'Modelo base\nIncluye días de hospitalización y atenciones',
        'Modelo sin variables operativas\nExcluye días de hospitalización y atenciones',
    ]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
        fig, ax = plt.subplots(figsize=(12, 6.4))
        fig.subplots_adjust(left=.38, right=.95, top=.78, bottom=.27)
        ax.barh(range(3), valores, height=.5, color=['#2879b8', '#259d76', '#d97935'])
        ax.invert_yaxis()
        ax.set_yticks(range(3), nombres)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel('R² en el conjunto de prueba (%)', labelpad=12)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#dde3e8', linewidth=.8)
        for lado in ['top', 'right', 'left']:
            ax.spines[lado].set_visible(False)
        ax.spines['bottom'].set_color('#bdc6ce')
        for i, valor in enumerate(valores):
            ax.text(valor + 1.8, i, f'{valor:.2f}%'.replace('.', ','), va='center', weight='bold', color='#263746')
        fig.text(.055, .94, 'Comparación de modelos: variabilidad del costo explicada', fontsize=16, weight='bold', color='#19384e')
        fig.text(.055, .885, 'Afiliados SIS · 570.437 registros procesados · Partición de entrenamiento/prueba: 80/20', fontsize=10.5, color='#526574')
        fig.text(.055, .13, 'El enriquecimiento con INEI mantiene el desempeño del modelo base.\nAl excluir las variables operativas, el R² baja de 73,93% a 1,78% (72,15 puntos porcentuales).', fontsize=11, color='#263746', linespacing=1.6)
        fig.text(.055, .035, 'Fuente: comparativa_modelos_regresion.csv. R² mide variabilidad explicada, no porcentaje de aciertos.\nComparación de modelos completos sobre VALOR_NETO procesado y winsorizado; no estima efectos causales.', fontsize=9, color='#526574', linespacing=1.5)
        for extension in ['png', 'svg']:
            fig.savefig(BASE / 'salidas' / f'sintesis_regresion_corregida.{extension}', dpi=220, facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    main()
