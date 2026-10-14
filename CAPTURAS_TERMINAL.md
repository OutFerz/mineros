# Una captura independiente por punto

Ejecutar desde `mineros` con el entorno virtual activado. No fotografiar toda la
salida conjunta: imprimir un punto, capturar solamente ese bloque y guardar el PNG
con el nombre indicado. Los comandos leen los CSV; no entrenan modelos.

1. `python resumen_terminal.py --seccion estadisticos` → `capturas/terminal_estadisticos.png`
2. `python resumen_terminal.py --seccion costo_edad` → `capturas/terminal_costo_edad.png`
3. `python resumen_terminal.py --seccion reglas` → `capturas/terminal_reglas.png`
4. `python resumen_terminal.py --seccion soporte_confianza` → `capturas/terminal_soporte_confianza.png`
5. `python resumen_terminal.py --seccion sucesion` → `capturas/terminal_sucesion.png`
6. `python resumen_terminal.py --seccion modelos` → `capturas/terminal_modelos.png`
7. `python resumen_terminal.py --seccion validacion` → `capturas/terminal_validacion.png`
8. `python resumen_terminal.py --seccion residuos` → `capturas/terminal_residuos.png`
9. `python resumen_terminal.py --seccion coeficientes` → `capturas/terminal_coeficientes.png`

Los nueve puntos ya tienen capturas independientes aportadas por el usuario.
También se incluye `terminal_mejor.png` junto a la interpretación de la regla con mayor lift.
Las últimas capturas incorporadas son sucesión, residuos y coeficientes (puntos 5, 8 y 9).
Los puntos 8 y 9 requieren los CSV nuevos `diagnostico_residuos.csv` y
`coeficientes_regresion.csv`, exportados al ejecutar el paso 8 de `completo.py`
actualizado. No se deben inventar estos valores a partir del PNG.

Después de guardar las capturas:

```bash
python actualizar_terminales.py
```

El script informa cuántas de las nueve capturas existen y coloca cada una encima
de su gráfico. No recorta la imagen conjunta ni repite las capturas originales.
Las capturas son estáticas; si cambian los CSV, deben renovarse.
