"""Sincroniza las secciones KNN del HTML con una ejecución completa existente."""
from pathlib import Path
import json
import re
import pandas as pd

BASE=Path(__file__).resolve().parent


def actualizar(salida=BASE/'resultados_e3_corregidos'):
    salida=Path(salida)
    meta=json.loads((salida/'metadatos_ejecucion.json').read_text(encoding='utf-8'))
    assert 'knn_desempate' in meta['clasificacion'], 'Se requieren resultados del KNN estable.'
    k=meta['clasificacion']['k_elegido']
    m=pd.read_csv(salida/'metricas_clasificacion.csv').set_index('modelo').loc['KNN']
    val=pd.read_csv(salida/'seleccion_k.csv')
    tn,fp,fn,tp=[n for row in json.loads((salida/'matriz_confusion_knn.json').read_text()) for n in row]
    assert tn+fp+fn+tp==meta['clasificacion']['prueba']
    f=lambda x,n=4:f'{x:.{n}f}'.replace('.',',')
    percent=lambda x:f(x*100,2)+'%'
    relative=salida.relative_to(BASE).as_posix()
    k_math=f'<math><mi>k</mi><mo>=</mo><mn>{k}</mn></math>'
    section=f'''<h3>3.1. Estandarización, desempate reproducible y selección de k</h3>
<p>La edad se estandariza y las categorías se codifican dentro del pipeline. K-NN mantiene ponderación inversa a la distancia. Para resolver vecinos equidistantes, se ordena por distancia euclidiana cuadrada redondeada a 12 decimales y luego por el identificador de fila de la muestra anterior a la partición. El desempate no utiliza la clase. Si existen vecinos a distancia cero, solo ellos votan; en empate de voto se elige la primera clase en orden. La tolerancia numérica forma parte del procedimiento documentado.</p>
<p>Se compararon k = 5, 11 y 21 exclusivamente en los 4.800 registros de validación interna. Se seleccionó {k_math} por F1 de la clase positiva: '''
    section+='; '.join(f'k={int(r.k)}: {f(r.f1_validacion)}' for _,r in val.iterrows())+'. No se usó el conjunto de prueba para esta decisión.</p>\n'
    section+=f'''<figure><img src="{relative}/seleccion_k.png" alt="Selección de k con desempate estable en validación"><figcaption><strong>Figura 3.1:</strong> Selección del número de vecinos mediante validación interna.</figcaption></figure>
<h3>3.2. Prueba independiente y desequilibrio de clases</h3>
<p>Con {k_math}, los 6.000 casos de prueba producen los siguientes resultados.</p>
<figure><img src="{relative}/confusion_knn.png" alt="Matriz de confusión KNN estable con k={k}"><figcaption><strong>Figura 3.2:</strong> Matriz de confusión de KNN con desempate estable.</figcaption></figure>
<ul><li><strong>Verdaderos negativos:</strong> {tn:,}; especificidad {percent(tn/(tn+fp))}.</li>
<li><strong>Falsos positivos:</strong> {fp:,}.</li>
<li><strong>Falsos negativos:</strong> {fn:,}; omisión del {percent(fn/(fn+tp))} de los positivos.</li>
<li><strong>Verdaderos positivos:</strong> {tp:,} de {tp+fn:,} casos con al menos dos atenciones.</li></ul>
<p><strong>Interpretación:</strong> Accuracy = {percent(m.accuracy)}, Balanced Accuracy = {percent(m.balanced_accuracy)}, precisión positiva = {percent(m.precision_1)}, recall = {percent(m.recall_1)}, F1 = {f(m.f1_1)} y ROC-AUC = {f(m.roc_auc)}. La accuracy sigue por debajo del clasificador de referencia (84,02%) y la sensibilidad es baja. Esta ejecución sustituye las métricas anteriores obtenidas sin una regla explícita de desempate; no se eligió para maximizar el resultado en prueba.</p>
'''
    path=BASE/'informe_ev_3.html'
    html=path.read_text(encoding='utf-8')
    html,n=re.subn(r'<h3>3\.1\..*?(?=<!-- 4\. REGRESIÓN LOGÍSTICA -->)',lambda _:section+'\n    ',html,flags=re.S)
    assert n==1,n
    row='<tr><td>K-Nearest Neighbors (k='+str(k)+')</td>'+''.join(f'<td>{f(m[col])}</td>' for col in ['accuracy','balanced_accuracy','precision_1','recall_1','f1_1','roc_auc'])+'</tr>'
    html,n=re.subn(r'<tr>\s*<td>K-Nearest Neighbors \(k=\d+\)</td>.*?</tr>',lambda _:row,html,flags=re.S)
    assert n==1,n
    html=re.sub(r'K-NN \(AUC = [0-9,]+\)', 'K-NN (AUC = '+f(m.roc_auc,3)+')', html)
    html=re.sub(r'K-NN muestra una sensibilidad marcadamente insuficiente \([0-9,]+%\)', 'K-NN muestra una sensibilidad baja ('+percent(m.recall_1)+')', html)
    conclusion=f'<li><strong>K-Nearest Neighbors (K-NN):</strong> Con desempate estable y k={k}, obtuvo Accuracy de {percent(m.accuracy)}, Recall de {percent(m.recall_1)}, F1 de {f(m.f1_1)} y ROC-AUC de {f(m.roc_auc)}. Su sensibilidad sigue siendo inferior a CART, C4.5 y regresión logística en esta prueba.</li>'
    html,n=re.subn(r'<li><strong>K-Nearest Neighbors \(K-NN\):</strong>.*?</li>',lambda _:conclusion,html,flags=re.S)
    assert n==1,n
    assert all((BASE/src).exists() for src in re.findall(r'<img[^>]+src="([^"]+)"',html))
    path.write_text(html,encoding='utf-8')
    print('HTML sincronizado con KNN estable:',m.to_dict())


if __name__=='__main__':
    actualizar()
