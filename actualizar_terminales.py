"""Inserta resultados de CSV con aspecto de terminal sobre sus gráficos en el HTML."""
from pathlib import Path
from html import escape
import re
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    modelos = pd.read_csv(BASE / 'salidas/comparativa_modelos_regresion.csv')
    reglas = pd.read_csv(BASE / 'salidas/reglas_asociacion_top_lift.csv')
    estadisticos = pd.read_csv(BASE / 'salidas/estadisticos_descriptivos.csv', index_col=0)
    metricas = []
    cv = []
    for _, fila in modelos.iterrows():
        metricas.append(f'{fila.iloc[0]}\n' + '-' * 45 +
                        f'\nR² Test : {fila.iloc[1]:.6f}\nRMSE    : {fila.iloc[2]:.6f} soles'
                        f'\nMAE     : {fila.iloc[3]:.6f} soles')
        cv.append(f'{fila.iloc[0]}\nR² CV   : {fila.iloc[4]}\nRMSE CV : {fila.iloc[5]}\nMAE CV  : {fila.iloc[6]}')
    ranking = []
    for i, fila in reglas.head(10).iterrows():
        ranking.append(f'R{i+1}: {fila["antecedents"]}\n -> {fila["consequents"]}\n'
                       f'Soporte: {fila["support"]:.2%} | Confianza: {fila["confidence"]:.2%} | Lift: {fila["lift"]:.6f}')
    descriptivos = estadisticos[['EDAD', 'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP']].to_string()
    bloques = {
        'histogramas_distribucion.png': ('Estadísticos descriptivos', descriptivos, 'estadisticos_descriptivos.csv'),
        'boxplots_diagnosticos.png': ('Resumen de costo y edad', estadisticos[['EDAD', 'VALOR_NETO']].loc[['min', '25%', '50%', '75%', 'max']].to_string(), 'estadisticos_descriptivos.csv'),
        'asociacion_top_lift.png': ('Reglas ordenadas por lift', '\n\n'.join(ranking), 'reglas_asociacion_top_lift.csv'),
        'asociacion_soporte_confianza.png': ('Soporte, confianza y lift', reglas[['support', 'confidence', 'lift']].head(10).rename_axis('Índice CSV (R1 = 0)').to_string(float_format=lambda x: f'{x:.6f}'), 'reglas_asociacion_top_lift.csv'),
        'sucesion_reglas_metricas.png': ('Sucesión ordenada de reglas', 'Orden: R1 a R10 según el CSV; no representa tiempo.\n\n' + reglas[['lift', 'confidence', 'support']].head(10).to_string(index=False), 'reglas_asociacion_top_lift.csv'),
        'regresion_comparativa_metricas.png': ('Resultados de los cinco modelos', '\n\n'.join(metricas), 'comparativa_modelos_regresion.csv'),
        'regresion_validacion_cruzada.png': ('Validación cruzada de cinco pliegues', '\n\n'.join(cv), 'comparativa_modelos_regresion.csv'),
        'regresion_predicciones_residuos.png': ('Evaluación del modelo enriquecido', metricas[1] + '\n\nEl gráfico muestra hasta 5.000 casos de prueba.\nLos CSV agregados no contienen las predicciones ni residuos individuales.', 'comparativa_modelos_regresion.csv'),
        'regresion_coeficientes.png': ('Contexto de los coeficientes', 'Modelo: Enriquecido (+ población INEI y tasa).\nCoeficientes por desviación estándar del predictor (soles).\nLos valores individuales de las barras no se exportan al CSV de métricas.\nConsultar las etiquetas del gráfico y la salida del paso 8.', 'comparativa_modelos_regresion.csv (identificación del modelo)'),
    }
    ruta = BASE / 'informe_mineria_datos.html'
    html = ruta.read_text(encoding='utf-8')
    html = re.sub(r'<!-- INICIO TERMINAL CSV -->.*?<!-- FIN TERMINAL CSV -->\n?', '', html, flags=re.S)
    def insertar(match):
        figura = match.group(0)
        nombre = Path(re.search(r'src="([^"]+)"', figura).group(1)).name
        if nombre not in bloques:
            return figura
        titulo, texto, fuente = bloques[nombre]
        texto = '\n'.join(linea.rstrip() for linea in texto.splitlines())
        return ('<!-- INICIO TERMINAL CSV -->\n'
                f'<aside class="terminal-resultados"><h4>Terminal — {escape(titulo)}</h4>'
                f'<p>Resumen generado desde salidas/{escape(fuente)}; no es una terminal interactiva.</p>'
                f'<pre>{escape(texto)}</pre></aside>\n<!-- FIN TERMINAL CSV -->\n' + figura)
    html, cantidad = re.subn(r'<figure>\s*<img\b[^>]+>.*?</figure>', insertar, html, flags=re.S)
    if '/* TERMINALES CSV */' not in html:
        html = html.replace('</style>', '''/* TERMINALES CSV */
        .terminal-resultados { background:#131c2b; color:#dce5f2; padding:20px; border-radius:10px; margin:20px 0 8px; }
        .terminal-resultados h4 { color:#85dfaa; margin:0 0 12px; }
        .terminal-resultados p { font-size:.8em; color:#bdcbdc; text-align:left; }
        .terminal-resultados pre { white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.6 Consolas,monospace; }
    </style>''', 1)
    assert html.count('<!-- INICIO TERMINAL CSV -->') == len(bloques)
    ruta.write_text(html, encoding='utf-8')
    print(f'Actualizados {len(bloques)} bloques de resultados; gráficos y modelos sin modificar.')


if __name__ == '__main__':
    main()
