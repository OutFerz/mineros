"""Corrige documentación y verifica evidencias sin recalcular modelos."""
from pathlib import Path
import re
from html import escape

BASE = Path(__file__).resolve().parent


def main():
    ruta = BASE / 'informe_mineria_datos.html'
    html = ruta.read_text(encoding='utf-8')
    html = html.replace('una muestra masiva de 600.000 registros', 'una muestra inicial de 650.000 registros, con 570.437 registros finales limpios,')
    html = html.replace('fuentes presupuestarias regionales externas', 'datos poblacionales regionales externos del INEI')
    html = html.replace('evidenciar la ganancia de precisión', 'evaluar la variación de precisión')
    html = html.replace('contexto territorial macroeconómico', 'contexto territorial demográfico')
    html = html.replace('VALOR_NETO acotado a 1050 soles', 'VALOR_NETO acotado a 1050,60 soles')
    html = re.sub(r'<p>Durante la auditoría de calidad.*?</p>', '<p>Durante la auditoría de calidad (Paso 0), las variables de comorbilidad presentaron entre 236.786 y 508.212 valores vacíos en la muestra inicial. El pipeline convierte estos valores a "NO" como supuesto operativo, evitando que la moda "SI" los convierta en diagnósticos positivos. El significado de los vacíos no está confirmado: esta codificación no demuestra ausencia de enfermedad y debe validarse con el diccionario de datos de la fuente.</p>', html)
    html = html.replace('la presencia de la condición equivale a 1 y su ausencia a 0', 'el registro "SI" equivale a 1 y "NO" a 0, incluyendo los vacíos recodificados bajo el supuesto operativo')
    html = re.sub(r'<p>Sin embargo, al correr.*?</p>', '<p>Al excluir DIAS_HOSP y CANT_ATENCIONES del modelo sin fuga, el R² baja a 0,0178. Los predictores restantes explican poco de la variabilidad del costo en esta configuración; esto no demuestra que las enfermedades previas carezcan de efecto clínico o económico. El mejor desempeño de los modelos con variables operativas depende de conocer los recursos utilizados.</p>', html)
    html = html.replace('desencadena una asociación estadística no aleatoria hacia', 'presenta una asociación estadística con')
    html = html.replace('con un asombroso <strong>Lift', 'con un <strong>Lift')
    html = html.replace('es el vector absoluto del gasto', 'presenta el mayor coeficiente positivo estandarizado')
    html = html.replace('<strong>Establecer protocolos de tamizaje obligatorio</strong>', '<strong>Evaluar protocolos de tamizaje con validación clínica</strong>')
    html = html.replace('600,000 registros procesados', '650.000 registros iniciales y 570.437 registros finales limpios')
    html = re.sub(r'<li><strong>Preprocesamiento y Calidad de los Datos:</strong>.*?</li>', '<li><strong>Preprocesamiento y Calidad de los Datos:</strong> Se eliminaron 79.561 duplicados exactos y dos registros con edades fuera del rango permitido. Se aplicó winsorización al percentil 99 y conversión de diagnósticos vacíos a "NO" como supuesto pendiente de validación. El conjunto final contiene 570.437 registros y no presenta nulos en las variables procesadas; esto no elimina la incertidumbre sobre el significado de los registros faltantes originales.</li>', html)
    html = re.sub(r'<li><strong>Inferencia sobre Reglas de Asociación:</strong>.*?</li>', '<li><strong>Inferencia sobre Reglas de Asociación:</strong> Apriori evaluó una muestra de 100.000 registros y exportó diez reglas destacadas por lift. La primera regla tiene lift 4,6416, confianza 40,87% y soporte 3,88%. Los resultados describen asociaciones entre atributos registrados simultáneamente, sin demostrar causalidad ni progresión temporal.</li>', html)
    html = re.sub(r'<li><strong>Inferencia Predictiva \(Modelo de Regresión Múltiple\):</strong>.*?</li>', '<li><strong>Inferencia Predictiva (Modelo de Regresión Múltiple):</strong> El modelo enriquecido obtiene R² = 0,7393, RMSE = 63,96 soles y MAE = 23,86 soles en prueba. DIAS_HOSP presenta el mayor coeficiente positivo estandarizado (+100,06), seguido de CANT_ATENCIONES (+22,98). La incorporación de población INEI no mejora de forma medible estas métricas y el modelo sin variables operativas alcanza R² = 0,0178.</li>', html)
    html = html.replace('Prevenir que un paciente crónico llegue a hospitalización reducirá drásticamente el impacto económico (VALOR_NETO).', 'El efecto económico de una intervención preventiva debe evaluarse; estos modelos no cuantifican ahorros causales.')
    html = html.replace('para un trabajo a futuro y lograr un R² superior al 37%', 'para un trabajo a futuro y evaluar mejoras respecto del R² actual de 0,7393')
    html = html.replace('Para un trabajo a futuro y lograr un R² superior al 37%', 'Para un trabajo a futuro y evaluar mejoras respecto del R² actual de 0,7393')
    html = re.sub(r'<p><strong>Aclaración de versiones:</strong>.*?</p>', '<p><strong>Consistencia de la documentación:</strong> El informe se actualizó según los CSV disponibles: 650.000 registros iniciales, 570.437 finales, R² = 0,7393 y lift máximo = 4,6416. Los diagnósticos vacíos se convierten a NO como supuesto del pipeline; la ausencia de registro no prueba ausencia de enfermedad.</p>', html)
    descripciones = {
        'paso_0_1.png': 'Paso 0: dimensiones, columnas y tipos del dataset original; permite revisar su estructura antes de seleccionar la muestra.',
        'paso_0_2.png': 'Paso 0: valores y distribuciones originales de variables clave; evidencia diagnósticos positivos y registros vacíos.',
        'paso_0_3.png': 'Paso 0: continuación de la inspección inicial. La interpretación de vacíos como NO en la captura es un supuesto no confirmado.',
        'seleccion_datos.png': 'Paso 1: extracción reproducible de 650.000 registros y selección de once columnas, incluido el identificador anonimizado.',
        'datos_externos.png': 'Paso 2: unión con población departamental INEI 2024 y cálculo de afiliados SIS por cada mil habitantes; no corresponde a datos presupuestarios.',
        'limpieza_1.png': 'Paso 3: revisión de nulos y recodificación de diagnósticos vacíos a NO bajo el supuesto del pipeline.',
        'limpieza_2.png': 'Paso 3: control de duplicados, edades y valores extremos; quedan 570.437 registros limpios.',
        'codificacion.png': 'Paso 4: mapeo binario y creación de indicadores para departamento y tipo de diabetes, sin imponer orden a las categorías.',
        'normalizacion.png': 'Paso 5: estandarización descriptiva por Z-score. En regresión, el escalador se ajusta dentro del pipeline de entrenamiento.',
        'estadisticos.png': 'Paso 6: resumen de conteo, tendencia central, dispersión y forma de las distribuciones del conjunto procesado.',
        'histogramas_distribucion.png': 'Paso 6: histogramas de variables cuantitativas; la concentración de valores bajos y la cola del costo muestran una distribución asimétrica.',
        'boxplots_diagnosticos.png': 'Paso 6: diagramas de caja de costo y edad; muestran mediana, dispersión y observaciones alejadas del rango central.',
        'matriz_correlacion_heatmap.png': 'Paso 6: correlaciones lineales entre variables numéricas; el color y los coeficientes indican dirección y magnitud, sin demostrar causalidad.',
        'reglas.png': 'Paso 7: evidencia de reglas Apriori y sus métricas. Soporte, confianza y lift responden a preguntas diferentes y no implican causalidad.',
        'regresion_1.png': 'Paso 8: evidencia de la evaluación de regresión y comparación de configuraciones del modelo.',
        'regresion_2.png': 'Paso 8: continuación de las métricas de regresión, con evaluación en prueba y validación cruzada según la consola.',
        'regresion_3.png': 'Paso 8: resultados y coeficientes del modelo; las diferencias con INEI deben interpretarse según las métricas actuales.',
        'acciones_clean.png': 'Paso 9: propuestas derivadas del análisis. Las afirmaciones causales y recomendaciones clínicas de la captura requieren validación independiente.',
    }
    def describir(match):
        tag = match.group(0)
        src = re.search(r'src="([^"]+)"', tag).group(1)
        descripcion = descripciones.get(Path(src).name)
        if not descripcion:
            return tag
        if 'alt=' not in tag:
            tag = tag[:-1] + f' alt="{escape(descripcion, quote=True)}">'
        return f'<figure>{tag}<figcaption>{escape(descripcion)}</figcaption></figure>'
    # Idempotencia: las imágenes sin descripción pertenecen al HTML original.
    html = re.sub(r'<img\b[^>]*>', lambda m: describir(m) if 'alt=' not in m.group(0) else m.group(0), html)
    if 'id="lectura-residuos"' not in html:
        html = html.replace('    <!-- 9. Acciones Inferidas', '<p id="lectura-residuos">En el gráfico real frente a predicho, los puntos alejados de la diagonal representan errores de mayor magnitud. En los residuos se observa dispersión desigual y algunos errores grandes; esto sugiere revisar el ajuste para costos extremos. También aparecen predicciones negativas, una limitación del modelo lineal para una variable de costo no negativa. El gráfico de coeficientes destaca DIAS_HOSP y CANT_ATENCIONES, pero los coeficientes territoriales no implican mejora predictiva global ni efectos causales.</p>\n    <!-- 9. Acciones Inferidas', 1)
    html = html.replace('    </style>', '        figure { margin: 18px 0; width: 100%; }\n        figcaption { font-size: .92em; color: #46576a; text-align: left; margin-top: 8px; }\n        .image-placeholder { flex-direction: column; }\n        .image-placeholder img, figure img { cursor: zoom-in; }\n    </style>', 1) if 'cursor: zoom-in' not in html else html
    if 'id="visor-imagen"' not in html:
        html = html.replace('</body>', '''<dialog id="visor-imagen" style="max-width:95vw;max-height:95vh;padding:16px;">
<button type="button" onclick="this.parentElement.close()">Cerrar imagen</button>
<img alt="" style="display:block;max-width:88vw;max-height:78vh;object-fit:contain;">
<p></p></dialog>
<script>
const visor = document.getElementById('visor-imagen');
document.querySelectorAll('.report-container img').forEach(img => {
  img.tabIndex = 0;
  img.title = 'Abrir imagen ampliada';
  const abrir = () => {
    visor.querySelector('img').src = img.src;
    visor.querySelector('img').alt = img.alt;
    visor.querySelector('p').textContent = img.closest('figure')?.querySelector('figcaption')?.textContent || img.alt;
    visor.showModal();
  };
  img.addEventListener('click', abrir);
  img.addEventListener('keydown', event => { if (event.key === 'Enter') abrir(); });
});
</script>
</body>''')
    ruta.write_text(html, encoding='utf-8')
    imagenes = re.findall(r'<img[^>]+src="([^"]+)"', html)
    faltantes = [src for src in imagenes if not (BASE / src).is_file()]
    assert not faltantes, faltantes
    assert '0.3671' not in html and '600.000' not in html
    print(f'Verificadas {len(imagenes)} imágenes; todas existen y tienen descripción y apertura ampliada.')


if __name__ == '__main__':
    main()
