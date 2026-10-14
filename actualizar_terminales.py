"""Incorpora la nueva captura de resumen CSV facilitada por el usuario.

La imagen es evidencia estática: este comando no toma capturas de pantalla.
Para imprimir datos actuales: python resumen_terminal.py
"""
from pathlib import Path
from html import escape
import re

BASE = Path(__file__).resolve().parent


def main():
    ruta = BASE / 'informe_mineria_datos.html'
    html = ruta.read_text(encoding='utf-8')
    for marcador in ['TERMINAL CSV', 'CAPTURA TERMINAL', 'RESUMEN CSV NUEVO']:
        html = re.sub(f'<!-- INICIO {marcador} -->.*?<!-- FIN {marcador} -->\\n?', '', html, flags=re.S)
    # Se conserva el PNG original; el HTML muestra cada sección en su lugar.
    zonas = {
        'histogramas_distribucion.png': (0, 242, 'Paso 6: resumen de estadísticos leído desde los CSV.'),
        'asociacion_top_lift.png': (1120, 320, 'Paso 7: diez reglas y sus métricas leídas desde el CSV. Entre reglas empatadas en lift, la captura puede presentar un orden distinto al CSV o al gráfico actual; R3/R4 y R7/R8 intercambian posición. Comparar también antecedentes y consecuentes.'),
        'regresion_comparativa_metricas.png': (246, 866, 'Paso 8: resultados y validación de los cinco modelos desde el CSV.'),
    }
    def insertar(match):
        figura = match.group(0)
        src = re.search(r'<img[^>]+src="([^"]+)"', figura)
        nombre = Path(src.group(1)).name if src else ''
        if nombre not in zonas:
            return figura
        y, alto, texto = zonas[nombre]
        descripcion = escape(texto, quote=True)
        captura = ('<!-- INICIO RESUMEN CSV NUEVO -->\n<figure>'
                   f'<div style="position:relative;aspect-ratio:891/{alto};overflow:hidden;background:#1e1e1e;">'
                   '<img src="capturas/resumen_csv_terminal.png" '
                   f'alt="{descripcion}" style="position:absolute;top:0;left:0;width:100%;max-width:100%;height:auto;transform:translateY(-{y/1600*100:.4f}%);">'
                   f'</div><figcaption>{descripcion} Captura aportada por el equipo; clic para ver la imagen completa. '
                   'Es evidencia estática y no se regenera al ejecutar el script.</figcaption></figure>\n'
                   '<!-- FIN RESUMEN CSV NUEVO -->\n')
        return captura + figura
    html = re.sub(r'<figure>.*?</figure>', insertar, html, flags=re.S)
    assert html.count('<!-- INICIO RESUMEN CSV NUEVO -->') == 3
    assert all((BASE / src).is_file() for src in re.findall(r'<img[^>]+src="([^"]+)"', html))
    ruta.write_text(html, encoding='utf-8')
    print('Incorporada la captura nueva: estadísticos, reglas y modelos en sus secciones. Quitadas las nueve repeticiones anteriores.')


if __name__ == '__main__':
    main()
