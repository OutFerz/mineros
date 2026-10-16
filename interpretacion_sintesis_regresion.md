# Interpretación complementaria de los resultados actuales

Los resultados corresponden a 570.437 registros procesados. El modelo enriquecido alcanza R² = 0,7393 en prueba y las reglas de asociación presentan un lift máximo de 4,642. Estas asociaciones no demuestran causalidad. La conversión de diagnósticos vacíos a «NO» es un supuesto del pipeline pendiente de validar con el diccionario de datos.

## Interpretación del gráfico

1. **Relevancia de las variables operativas.** Los modelos base y enriquecido explican el **73,93% de la variabilidad del costo** en prueba. Al excluir días de hospitalización y cantidad de atenciones, el R² baja a **1,78%**, una diferencia de **72,15 puntos porcentuales**. La comparación muestra cuánto depende el desempeño de disponer de esas variables; no mide el efecto causal de hospitalizar a un paciente.

2. **Limitación para anticipar el gasto.** El modelo sin variables operativas conserva diagnósticos, información demográfica y variables territoriales, pero presenta baja capacidad explicativa. Esto limita su utilidad para anticipar costos con los predictores restantes; no demuestra que las enfermedades crónicas solo generen gasto cuando hay internamiento.

3. **Aporte de la información INEI.** El enriquecimiento mantiene prácticamente el mismo R² y RMSE que el modelo base. En esta evaluación, la población regional y la tasa de afiliados no aportan una mejora predictiva medible. Pueden aportar contexto a la planificación territorial, aunque ese uso no fue evaluado por estos modelos de costo individual.

El R² expresa variabilidad explicada, no porcentaje de predicciones correctas. Los resultados se refieren al costo procesado y winsorizado; requieren validación temporal y territorial antes de generalizarlos.
