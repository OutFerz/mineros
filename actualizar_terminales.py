"""Coloca una captura independiente por gráfico; informa las que faltan.

Imprimir un punto: python resumen_terminal.py --seccion costo_edad
Guardar la captura auténtica en capturas/terminal_costo_edad.png y ejecutar este script.
"""
from pathlib import Path
from html import escape
import re

BASE = Path(__file__).resolve().parent
PUNTOS = {
    'histogramas_distribucion.png': ('estadisticos', 'Estadísticos descriptivos'),
    'boxplots_diagnosticos.png': ('costo_edad', 'Resumen de costo y edad'),
    'asociacion_top_lift.png': ('reglas', 'Reglas ordenadas por lift'),
    'asociacion_soporte_confianza.png': ('soporte_confianza', 'Soporte, confianza y lift'),
    'sucesion_reglas_metricas.png': ('sucesion', 'Sucesión ordenada de reglas'),
    'regresion_comparativa_metricas.png': ('modelos', 'Comparación de modelos'),
    'regresion_validacion_cruzada.png': ('validacion', 'Validación cruzada'),
    'regresion_predicciones_residuos.png': ('residuos', 'Diagnóstico de residuos'),
    'regresion_coeficientes.png': ('coeficientes', 'Coeficientes del modelo'),
}


def main():
    ruta = BASE / 'informe_mineria_datos.html'
    html = ruta.read_text(encoding='utf-8')
    for marcador in ['TERMINAL CSV', 'CAPTURA TERMINAL', 'RESUMEN CSV NUEVO', 'CAPTURA INDIVIDUAL']:
        html = re.sub(f'<!-- INICIO {marcador} -->.*?<!-- FIN {marcador} -->\\n?', '', html, flags=re.S)
    presentes, faltantes = [], []
    def insertar(match):
        figura = match.group(0)
        src = re.search(r'<img[^>]+src="([^"]+)"', figura)
        nombre = Path(src.group(1)).name if src else ''
        if nombre not in PUNTOS:
            return figura
        punto, descripcion = PUNTOS[nombre]
        captura = f'capturas/terminal_{punto}.png'
        if not (BASE / captura).is_file():
            faltantes.append((punto, captura))
            return figura
        presentes.append(punto)
        return ('<!-- INICIO CAPTURA INDIVIDUAL -->\n'
                f'<figure><img src="{captura}" alt="Captura de consola: {escape(descripcion)}" style="max-width:100%;height:auto;">'
                f'<figcaption>Captura independiente: {escape(descripcion)}.</figcaption></figure>\n'
                '<!-- FIN CAPTURA INDIVIDUAL -->\n' + figura)
    html = re.sub(r'<figure>.*?</figure>', insertar, html, flags=re.S)
    assert len(presentes) + len(faltantes) == 9
    ruta.write_text(html, encoding='utf-8')
    print(f'Capturas independientes incorporadas: {len(presentes)}/9')
    for punto, captura in faltantes:
        print(f'PENDIENTE: python resumen_terminal.py --seccion {punto} -> guardar captura en {captura}')


if __name__ == '__main__':
    main()
