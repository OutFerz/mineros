"""Añade evidencia al HTML conservando íntegro el contenido anterior."""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent


def figura(nombre, texto):
    return (f'<figure><img src="salidas/{nombre}" alt="{texto}" '
            f'style="max-width:100%;height:auto;"><figcaption>{texto}</figcaption></figure>\n')


def main():
    ruta = BASE / 'informe_mineria_datos.html'
    original = ruta.read_text(encoding='utf-8')
    if 'id="interpretacion-resultados-actuales"' in original:
        print('La ampliación ya está incorporada.')
        return
    modelos = pd.read_csv(BASE / 'salidas/comparativa_modelos_regresion.csv')
    reglas = pd.read_csv(BASE / 'salidas/reglas_asociacion_top_lift.csv')
    b, e, ridge, sf, log = [modelos.iloc[i] for i in range(5)]
    r = reglas.iloc[0]
    asociacion = '\n<!-- INICIO ADICION ASOCIACION -->\n<h3>7.1. Resultados gráficos de asociación y sucesión ordenada</h3>\n'
    asociacion += figura('asociacion_top_lift.png', 'Ranking de reglas según lift; R1 a R10 corresponden al CSV de reglas.')
    asociacion += figura('asociacion_soporte_confianza.png', 'Relación entre soporte y confianza; el color representa el lift.')
    asociacion += figura('sucesion_reglas_metricas.png', 'Sucesión de métricas en el orden de las reglas, sin significado temporal.')
    asociacion += (f'<p>La primera regla presenta soporte de <strong>{r["support"]:.2%}</strong>, confianza de '
                    f'<strong>{r["confidence"]:.2%}</strong> y lift de <strong>{r["lift"]:.3f}</strong>. '
                    'El soporte mide la proporción de registros que cumplen conjuntamente antecedente y consecuente; '
                    'la confianza mide la proporción con el consecuente entre quienes cumplen el antecedente. '
                    'El lift compara esa confianza con la frecuencia general del consecuente. Un lift alto '
                    'no implica causalidad ni garantiza que la mayoría de los pacientes presente el patrón. '
                    'Las reglas inversas pueden compartir soporte y lift, pero tener distinta confianza.</p>\n'
                    '<p>La sucesión representa un ranking descriptivo. No se analizó una secuencia de eventos '
                    'clínicos: para estudiar progresión temporal se requieren fechas y eventos ordenados por paciente. '
                    'Estos gráficos muestran únicamente las diez reglas exportadas, no todas las reglas descubiertas.</p>\n<!-- FIN ADICION ASOCIACION -->\n')
    regresion = '\n<!-- INICIO ADICION REGRESION -->\n<h3>8.1. Comparación gráfica y análisis de regresión</h3>\n'
    regresion += figura('regresion_comparativa_metricas.png', 'Comparación de R², RMSE y MAE de los cinco modelos sobre la partición de prueba.')
    regresion += figura('regresion_validacion_cruzada.png', 'Media y desviación estándar de R² en validación cruzada de cinco pliegues; no son intervalos de confianza.')
    regresion += (f'<p>El modelo enriquecido alcanza R² = <strong>{e.iloc[1]:.4f}</strong>, '
                   f'RMSE = <strong>{e.iloc[2]:.2f} soles</strong> y MAE = <strong>{e.iloc[3]:.2f} soles</strong>. '
                   'El R² resume la fracción de variabilidad explicada en prueba; el MAE representa el error absoluto '
                   'medio, mientras que el RMSE penaliza más los errores grandes. La distancia entre RMSE y MAE '
                   'es compatible con errores de mayor magnitud en una parte de los casos.</p>\n'
                   f'<p>Al excluir días de hospitalización y cantidad de atenciones, el R² baja a '
                   f'<strong>{sf.iloc[1]:.4f}</strong> y el RMSE sube a <strong>{sf.iloc[2]:.2f} soles</strong>. '
                   'Esta configuración explica poco del gasto con los predictores disponibles antes de la atención; '
                   'no demuestra que las comorbilidades carezcan de efecto clínico. Base, enriquecido y Ridge '
                   'presentan resultados prácticamente iguales. La información INEI no aporta una mejora medible '
                   'en esta comparación individual.</p>\n'
                   f'<p>El modelo logarítmico reconvertido obtiene R² = <strong>{log.iloc[1]:.4f}</strong> '
                   f'y RMSE = <strong>{log.iloc[2]:.2f} soles</strong>. Su R² negativo indica que su error cuadrático '
                   'supera el de predecir la media del conjunto evaluado; esta transformación no mejora los resultados '
                   'en la escala monetaria. Los gráficos de predicciones, residuos y coeficientes se generan '
                   'adicionalmente al ejecutar el pipeline completo, usando las predicciones reales, sin reconstruirlas '
                   'a partir de métricas agregadas.</p>\n<!-- FIN ADICION REGRESION -->\n')
    conclusion = '\n<!-- INICIO ADICION CONCLUSION -->\n<h3 id="interpretacion-resultados-actuales">Interpretación complementaria de los resultados actuales</h3>\n'
    conclusion += (f'<p><strong>Aclaración de versiones:</strong> se conserva el texto anterior solicitado, pero '
                    'sus referencias a R² = 0.3671, coeficiente 881.5 y ausencia de reglas corresponden a una '
                    'versión previa y no describen los CSV actuales. Para interpretar la ejecución disponible '
                    f'deben utilizarse R² = {e.iloc[1]:.4f} y las reglas con lift máximo de {r["lift"]:.3f}. '
                    'Además, la etapa de limpieza actual convierte los diagnósticos nulos a NO según la convención '
                    'asumida por el pipeline, en vez de imputarlos por moda; esta convención requiere validación '
                    'con el diccionario de datos porque la ausencia de registro no prueba ausencia de enfermedad.</p>\n'
                    f'<p><strong>Interpretación del resultado de regresión:</strong> el modelo enriquecido explica '
                    f'aproximadamente {e.iloc[1]*100:.2f}% de la variabilidad del gasto en prueba, sobre el costo '
                    'winsorizado por el procesamiento. Es útil para describir costos cuando ya se conocen los '
                    'recursos utilizados. Su desempeño no se puede trasladar a una predicción previa a la '
                    f'hospitalización, donde la configuración sin fuga explica solo {sf.iloc[1]*100:.2f}%. '
                    'Los coeficientes estandarizados expresan cambios asociados a una desviación estándar del '
                    'predictor, manteniendo los demás constantes; no equivalen al costo causal de un día adicional.</p>\n'
                    f'<p><strong>Interpretación de asociación:</strong> la combinación Obesidad, Adulto Mayor y '
                    'Costo Alto se asocia con Hipertensión, Mujer, Atenciones Frecuentes y DM Tipo 2. '
                    f'La confianza de {r["confidence"]:.2%} equivale a unos {r["confidence"]*100:.0f} de cada 100 '
                    f'registros que cumplen el antecedente, y el soporte de {r["support"]:.2%} describe su alcance '
                    'en la muestra de Apriori. Es una asociación entre atributos simultáneos, no una sucesión '
                    'de diagnósticos ni evidencia de que una condición desencadene otra.</p>\n'
                    '<p><strong>Aplicación y límites:</strong> los hallazgos apoyan estudiar perfiles de mayor '
                    'demanda de recursos y evaluar estrategias de seguimiento. No permiten cuantificar ahorros '
                    'causales ni justificar protocolos obligatorios por sí solos. Se requiere validación temporal '
                    'y territorial, revisar la disponibilidad de variables al momento de predecir y evaluar '
                    'los errores antes de usar el modelo en planificación. La estabilidad en cinco pliegues '
                    'describe la muestra analizada y no garantiza generalización a otros períodos.</p>\n<!-- FIN ADICION CONCLUSION -->\n')
    nuevo = original.replace('    <!-- 8. Aplicación', asociacion + '    <!-- 8. Aplicación', 1)
    nuevo = nuevo.replace('    <!-- 9. Acciones', regresion + '    <!-- 9. Acciones', 1)
    nuevo = nuevo.replace('\n</div>\n\n</body>', conclusion + '\n</div>\n\n</body>', 1)
    assert nuevo.replace(asociacion, '').replace(regresion, '').replace(conclusion, '') == original
    assert all((BASE / 'salidas' / nombre).exists() for nombre in [
        'asociacion_top_lift.png', 'asociacion_soporte_confianza.png',
        'sucesion_reglas_metricas.png', 'regresion_comparativa_metricas.png', 'regresion_validacion_cruzada.png'])
    ruta.write_text(nuevo, encoding='utf-8')
    print('HTML ampliado. Verificado: todo el contenido original permanece íntegro.')


if __name__ == '__main__':
    main()
