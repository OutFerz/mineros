"""Regenera diez gráficos complementarios desde resultados existentes y actualiza el HTML."""
from pathlib import Path
from html import escape
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SALIDAS = BASE / 'salidas'


def main():
    modelos = pd.read_csv(SALIDAS / 'comparativa_modelos_regresion.csv')
    reglas = pd.read_csv(SALIDAS / 'reglas_asociacion_top_lift.csv').head(10).reset_index(drop=True)
    estad = pd.read_csv(SALIDAS / 'estadisticos_descriptivos.csv', index_col=0)
    nombres = ['Base (SIS)', 'Enriquecido (INEI)', 'Ridge (+ INEI)', 'Sin variables operativas', 'Log1p reconvertido']
    figuras = []

    def guardar(fig, ax, numero, titulo, subtitulo, descripcion):
        if numero in [2, 4]:
            # Versión compacta: conserva métricas y explicita las variables excluidas.
            fig.suptitle(titulo, fontsize=15, weight='bold', y=.98)
            fig.text(.5, .925, subtitulo, ha='center', fontsize=10)
            fig.subplots_adjust(left=.25, right=.97, top=.87, bottom=.16 if numero == 4 else .11)
            if numero == 4:
                fig.text(.25, .025, 'Modelo log1p reconvertido: validación cruzada no reportada (N/A).', fontsize=10)
            for lado in ['top', 'right']:
                ax.spines[lado].set_visible(False)
            nombre = f'complementario_{numero:02d}.png'
            fig.savefig(SALIDAS / nombre, dpi=180, facecolor='white')
            plt.close(fig)
            figuras.append((numero, nombre, titulo, descripcion))
            return
        fig.suptitle(titulo, fontsize=16, weight='bold', y=.97)
        fig.text(.5, .9, subtitulo, ha='center', fontsize=10, color='#526574')
        fig.tight_layout(rect=[0, 0, 1, .86])
        nombre = f'complementario_{numero:02d}.png'
        fig.savefig(SALIDAS / nombre, dpi=180, facecolor='white')
        plt.close(fig)
        figuras.append((numero, nombre, titulo, descripcion))

    def barras(valores, etiquetas, eje, formato='.2f'):
        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax.barh(etiquetas, valores, color='#247eaf', height=.58)
        ax.invert_yaxis()
        ax.axvline(0, color='#8195a5', linewidth=.8)
        ax.set_xlabel(eje)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.2)
        ax.margins(x=.18)
        for i, v in enumerate(valores):
            ax.annotate(format(v, formato), (v, i), xytext=(6 if v >= 0 else -6, 0),
                        textcoords='offset points', ha='left' if v >= 0 else 'right', va='center', weight='bold')
        return fig, ax

    for numero, col, titulo, eje, desc, formato in [
        (1, 1, 'Regresión — R² por modelo', 'R² en prueba (puede ser negativo)', 'Base, enriquecido y Ridge presentan resultados similares; el modelo logarítmico reconvertido obtiene un R² negativo.', '.4f'),
        (2, 2, 'Regresión — raíz del error cuadrático medio (RMSE)', 'RMSE (soles) · menor es mejor', 'Base, enriquecido y Ridge alcanzan el menor RMSE, aproximadamente 63,96 soles.', '.2f'),
        (3, 3, 'Regresión — error absoluto medio (MAE)', 'MAE (soles) · menor es mejor', 'Los tres primeros modelos presentan un error absoluto medio de 23,86 soles.', '.2f'),
    ]:
        fig, ax = barras(modelos.iloc[:, col].to_numpy(), nombres, eje, formato)
        guardar(fig, ax, numero, titulo, 'Prueba 80/20 · fuente: comparativa_modelos_regresion.csv', desc)

    medias, desv = np.array([list(map(float, v.split('±'))) for v in modelos.iloc[:4, 4]]).T
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.barh(nombres[:4], medias, xerr=desv, capsize=4, color='#247eaf')
    ax.invert_yaxis()
    ax.set(xlabel='R² medio en validación cruzada', xlim=(0, .95))
    for i, (media, d) in enumerate(zip(medias, desv)):
        ax.text(media + .015, i, f'{media:.4f} ± {d:.4f}', va='center')
    guardar(fig, ax, 4, 'Regresión — estabilidad del R² en validación cruzada',
            '5 pliegues · media ± desviación estándar',
            'Base, enriquecido y Ridge muestran poca variación entre pliegues; las barras de error representan desviaciones estándar.')

    x = np.arange(1, len(reglas) + 1)
    for numero, col, titulo, eje, factor, desc in [
        (5, 'support', 'Apriori — soporte por regla', 'Soporte (%)', 100, 'Las combinaciones de cada regla aparecen conjuntamente en aproximadamente 3,88% a 5,49% de la muestra.'),
        (6, 'confidence', 'Apriori — confianza por regla', 'Confianza (%)', 100, 'La frecuencia del consecuente entre quienes cumplen el antecedente varía entre 29,51% y 60,41%.'),
        (7, 'lift', 'Apriori — lift por regla', 'Lift (escala ampliada)', 1, 'Todas las reglas muestran asociación positiva; las diferencias son pequeñas y el eje vertical está ampliado.'),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        valores = reglas[col].to_numpy() * factor
        if col == 'lift':
            ax.plot(x, valores, 'o-', color='#247eaf')
            ax.set_ylim(valores.min() - .015, valores.max() + .015)
        else:
            ax.bar(x, valores, color='#247eaf', width=.65)
            ax.set_ylim(0, valores.max() * 1.22)
        for pos, v in zip(x, valores):
            ax.annotate(f'{v:.4f}' if col == 'lift' else f'{v:.2f}', (pos, v), xytext=(0, 8), textcoords='offset points', ha='center')
        ax.set(xticks=x, xticklabels=[f'R{i}' for i in x], ylabel=eje, xlabel='Posición en el CSV actual, ordenado por lift (no representa tiempo)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        guardar(fig, ax, numero, titulo, 'Mismo orden R1–R10 del archivo reglas_asociacion_top_lift.csv', desc)

    fig, ax = barras([650000, estad.loc['count', 'EDAD']], ['Muestra inicial', 'Registros finales'], 'Cantidad de registros', ',.0f')
    guardar(fig, ax, 8, 'Dataset — registros iniciales y finales', 'Muestra inicial documentada: 650.000 · conteo final desde el CSV descriptivo', 'Después de la limpieza se conservan 570.437 de los 650.000 registros seleccionados.')
    edad = estad.loc[['min', '25%', '50%', 'mean', '75%', 'max'], 'EDAD'].to_numpy()
    fig, ax = barras(edad, ['Mínimo', 'Percentil 25', 'Mediana', 'Media', 'Percentil 75', 'Máximo'], 'Edad (años)')
    guardar(fig, ax, 9, 'Dataset — resumen de edad', 'Fuente: estadisticos_descriptivos.csv', 'La mediana es de 57 años y el 50% central se encuentra entre 46 y 68 años.')
    costo = estad.loc[['mean', 'std', 'max'], 'VALOR_NETO'].to_numpy()
    fig, ax = barras(costo, ['Costo promedio', 'Desviación estándar', 'Máximo procesado'], 'Soles (medidas descriptivas distintas)')
    guardar(fig, ax, 10, 'Dataset — resumen del costo procesado', 'VALOR_NETO después de winsorización · mediana y percentil 75: 0 soles', 'El costo medio es 25,86 soles, con alta dispersión y un máximo winsorizado de 1.050,60 soles.')

    html_path = BASE / 'informe_mineria_datos.html'
    html = html_path.read_text(encoding='utf-8')
    html = re.sub(r'<!-- INICIO DIEZ COMPLEMENTARIOS -->.*?<!-- FIN DIEZ COMPLEMENTARIOS -->\n?', '', html, flags=re.S)
    bloque = '<!-- INICIO DIEZ COMPLEMENTARIOS -->\n<h3>Gráficos complementarios de los resultados</h3>\n'
    for numero, nombre, titulo, descripcion in figuras:
        if numero in [1, 5, 8]:
            bloque += '<h4>' + {1: 'Regresión', 5: 'Reglas de asociación', 8: 'Dataset procesado'}[numero] + '</h4>\n'
        bloque += f'<figure><img src="salidas/{nombre}" alt="{escape(titulo)}" style="max-width:100%;height:auto;"><figcaption>{numero}. {escape(descripcion)}</figcaption></figure>\n'
    bloque += '<!-- FIN DIEZ COMPLEMENTARIOS -->\n'
    assert '<!-- FIN ADICION CONCLUSION -->' in html
    html = html.replace('<!-- FIN ADICION CONCLUSION -->', bloque + '<!-- FIN ADICION CONCLUSION -->', 1)
    assert all((BASE / src).is_file() for src in re.findall(r'<img[^>]+src="([^"]+)"', html))
    html_path.write_text(html, encoding='utf-8')
    print('Diez gráficos corregidos y añadidos debajo de la síntesis y su interpretación.')


if __name__ == '__main__':
    main()
