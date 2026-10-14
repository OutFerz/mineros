"""Imprime evidencia actual desde los CSV, sin ajustar modelos.

Uso: python resumen_terminal.py [--seccion estadisticos|modelos|reglas|mejor|todas]
"""
from pathlib import Path
import argparse
import pandas as pd

BASE = Path(__file__).resolve().parent


def titulo(texto):
    print('\n' + '=' * 66)
    print('  ' + texto)
    print('=' * 66)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seccion', choices=['estadisticos', 'modelos', 'reglas', 'mejor', 'todas'], default='todas')
    args = parser.parse_args()
    modelos = pd.read_csv(BASE / 'salidas/comparativa_modelos_regresion.csv')
    reglas = pd.read_csv(BASE / 'salidas/reglas_asociacion_top_lift.csv')
    estad = pd.read_csv(BASE / 'salidas/estadisticos_descriptivos.csv', index_col=0)
    if args.seccion in ['estadisticos', 'todas']:
        titulo('01 | ESTADISTICOS DESCRIPTIVOS')
        for nombre, columna, fila in [
            ('Registros analizados', 'EDAD', 'count'), ('Edad promedio', 'EDAD', 'mean'),
            ('Edad mediana', 'EDAD', '50%'), ('Costo promedio', 'VALOR_NETO', 'mean'),
            ('Desviacion del costo', 'VALOR_NETO', 'std'), ('Varianza del costo', 'VALOR_NETO', 'varianza'),
            ('Maximo costo procesado', 'VALOR_NETO', 'max')]:
            print(f'  {nombre:<30} {estad.loc[fila, columna]:>12,.2f}')
    if args.seccion in ['modelos', 'todas']:
        titulo('02 | MODELOS DE REGRESION')
        for _, r in modelos.iterrows():
            print(f'\n  {r.iloc[0]}\n  ' + '-' * 55)
            print(f'  R2 Test          : {r.iloc[1]:.6f}')
            print(f'  RMSE             : {r.iloc[2]:.6f}')
            print(f'  MAE              : {r.iloc[3]:.6f}')
            print(f'  Validacion R2 CV : {r.iloc[4]}')
    if args.seccion in ['reglas', 'todas']:
        titulo('03 | REGLAS DE ASOCIACION - TOP 10')
        print(f'  {"REGLA":<9} {"SOPORTE":>12} {"CONFIANZA":>14} {"LIFT":>12}\n  ' + '-' * 54)
        for i, r in reglas.head(10).reset_index(drop=True).iterrows():
            print(f'  R{i+1:<8} {r["support"]*100:>11.2f}% {r["confidence"]*100:>13.2f}% {r["lift"]:>12.4f}')
    if args.seccion in ['mejor', 'todas']:
        titulo('04 | REGLA CON MAYOR LIFT')
        r = reglas.loc[reglas['lift'].idxmax()]
        print(f'  Antecedentes:\n  {r["antecedents"]}\n\n  Consecuentes:\n  {r["consequents"]}')
        print(f'\n  Soporte          : {r["support"]*100:.3f}%')
        print(f'  Confianza        : {r["confidence"]*100:.3f}%')
        print(f'  Lift             : {r["lift"]:.6f}')
    titulo('VERIFICACION FINAL')
    print('  Datos cargados directamente desde los CSV del proyecto.')
    print('  No se recalcularon ni modificaron los modelos.')
    print('=' * 66)


if __name__ == '__main__':
    main()
