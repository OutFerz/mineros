"""Evaluación 3: Minería de datos - afiliados SIS con diabetes.

Uso:
    python analisis_evaluacion3.py --csv "Afiliados_activos_DM_SIS.csv"
    python analisis_evaluacion3.py --csv "archivo.csv" --salida resultados --n-clas 30000 --n-cluster 20000

No incluye el CSV. Los resultados solo son válidos tras la ejecución con los datos auténticos.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
REQ = ['EDAD','SEXO','DEPARTAMENTO','TIPO_DIABETES','CON_DX_OBESIDAD',
       'CON_DX_HIPERTENSION','CON_DX_SALUDMENTAL','CANT_ATENCIONES']
NUM = ['EDAD']
CAT = ['SEXO','DEPARTAMENTO','TIPO_DIABETES','CON_DX_OBESIDAD',
       'CON_DX_HIPERTENSION','CON_DX_SALUDMENTAL']
FEATURES = NUM + CAT
CLUSTER_NUM = ['EDAD','CANT_ATENCIONES','DIAS_HOSP','VALOR_NETO']
CLUSTER_CAT = ['SEXO','TIPO_DIABETES','CON_DX_OBESIDAD','CON_DX_HIPERTENSION',
               'CON_DX_SALUDMENTAL']


def limpiar_cabecera(c):
    c = unicodedata.normalize('NFKD',str(c)).encode('ascii','ignore').decode()
    return c.strip().upper().replace(' ','_').replace('-','_')


def leer_csv(path):
    """Lee CSV de distintas codificaciones / separadores sin alterar filas."""
    errores=[]
    for enc in ['utf-8-sig','latin-1']:
        for sep in [',',';','\t']:
            try:
                df = pd.read_csv(path,sep=sep,encoding=enc,low_memory=False)
                df.columns = [limpiar_cabecera(c) for c in df.columns]
                if len(set(REQ).intersection(df.columns))>=6:
                    return df
            except (UnicodeError,pd.errors.ParserError,ValueError) as exc:
                errores.append(f'{enc}, {sep}: {exc}')
    raise ValueError('No se reconocieron las columnas esperadas. Comprueba el CSV original y su separador. ' + '; '.join(errores[:2]))


def normaliza_texto(s):
    return s.astype('string').str.strip().str.upper().str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('ascii').replace({'':pd.NA,'NAN':pd.NA,'NONE':pd.NA,'NULL':pd.NA})


def muestrear_estratificado(df,n,col):
    if len(df)<=n:
        return df.copy().reset_index(drop=True)
    # Para comparación justa, seleccionar la misma muestra para TODOS los modelos.
    try:
        elegido,_=train_test_split(df,train_size=n,stratify=df[col],random_state=SEED)
    except ValueError as exc:
        raise ValueError(f'No se puede construir una muestra estratificada: {exc}')
    return elegido.reset_index(drop=True)


def preparar(path,n_base,comparar_e2):
    raw=leer_csv(path)
    missing=[c for c in REQ if c not in raw]
    if missing:
        raise ValueError(f'Faltan columnas del informe E2: {missing}. Encontradas: {raw.columns.tolist()}')
    originales=len(raw)
    # La E2 usa semilla fija pero NO se dispone del script ni de la semilla exacta.
    # Esta extracción reproduce el tamaño y fija una semilla DOCUMENTADA (42).
    if len(raw)>n_base:
        df=raw.sample(n=n_base,random_state=SEED).copy()
    else:
        df=raw.copy()
    muestra=len(df)
    # El informe E2 indica eliminación de registros duplicados exactos del mismo
    # paciente. NO es equivalente a dejar una sola fila por CODIGO_ANONIMIZADO:
    # una persona puede tener varias filas con atributos diferentes.
    # Se compara la misma selección de once columnas descritas en el informe E2.
    clave_duplicados = [c for c in [
        'CODIGO_ANONIMIZADO', 'EDAD', 'SEXO', 'DEPARTAMENTO',
        'TIPO_DIABETES', 'CON_DX_OBESIDAD', 'CON_DX_HIPERTENSION',
        'CON_DX_SALUDMENTAL', 'CANT_ATENCIONES', 'VALOR_NETO', 'DIAS_HOSP'
    ] if c in df.columns]
    if 'CODIGO_ANONIMIZADO' in df.columns:
        df = df.drop_duplicates(subset=clave_duplicados)
    else:
        print('AVISO: sin CODIGO_ANONIMIZADO; deduplicación exacta sin esa clave.')
        df = df.drop_duplicates(subset=clave_duplicados)
    post_dup=len(df)
    for c in ['EDAD','CANT_ATENCIONES','DIAS_HOSP','VALOR_NETO']:
        if c in df:
            v = df[c].astype('string').str.replace(',','.',regex=False)
            df[c]=pd.to_numeric(v,errors='coerce')
    df=df.loc[df['EDAD'].between(0,110) & df['CANT_ATENCIONES'].notna()].copy()
    validas=len(df)
    for c in CAT:
        df[c]=normaliza_texto(df[c])
    df['SEXO']=df['SEXO'].replace({'F':'FEMENINO','M':'MASCULINO'})
    for c in ['CON_DX_OBESIDAD','CON_DX_HIPERTENSION','CON_DX_SALUDMENTAL']:
        if comparar_e2:
            # Supuesto original E2; NO es evidencia de ausencia de diagnóstico.
            df[c]=df[c].fillna('NO')
        else:
            df[c]=df[c].fillna('SIN_REGISTRO')
    df['TIPO_DIABETES']=df['TIPO_DIABETES'].fillna('SIN_REGISTRO')
    df['DEPARTAMENTO']=df['DEPARTAMENTO'].fillna('SIN_REGISTRO')
    df['SEXO']=df['SEXO'].fillna('SIN_REGISTRO')
    # Criterio de esta evaluación, distinto de la definición no documentada en E2.
    df['ATENCIONES_FRECUENTES_E3']=(df['CANT_ATENCIONES']>=2).astype(int)
    distrib=df['ATENCIONES_FRECUENTES_E3'].value_counts().reindex([0,1],fill_value=0)
    if distrib.min() == 0:
        raise ValueError('El criterio >=2 atenciones produce una sola clase en este archivo; revisa el campo.')
    print(f'Archivo: {originales:,}; muestra inicial: {muestra:,}; tras duplicados exactos: {post_dup:,}; tras validación: {validas:,}')
    print('Clase 0/1:', distrib.to_dict())
    return df,{'originales':originales,'muestra_inicial':muestra,'tras_deduplicacion':post_dup,
               'tras_validacion':validas,'objetivo_0':int(distrib[0]),'objetivo_1':int(distrib[1]),
               'comparar_e2':bool(comparar_e2)}


def preprocess():
    return ColumnTransformer([
        ('num',Pipeline([('imputar',SimpleImputer(strategy='median')),
                         ('escalar',StandardScaler())]),NUM),
        ('cat',Pipeline([('imputar',SimpleImputer(strategy='most_frequent')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])


def guardar_fig(path):
    plt.tight_layout()
    plt.savefig(path,dpi=170,bbox_inches='tight')
    plt.close()


def evaluar(nombre,pipe,Xtr,ytr,Xte,yte,salida):
    pipe.fit(Xtr,ytr)
    p=pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:,1]
    d={'modelo':nombre,'accuracy':accuracy_score(yte,p),
       'balanced_accuracy':balanced_accuracy_score(yte,p),
       'precision_1':precision_score(yte,p,zero_division=0),
       'recall_1':recall_score(yte,p,zero_division=0),
       'f1_1':f1_score(yte,p,zero_division=0),
       'roc_auc':roc_auc_score(yte,proba)}
    fig,ax=plt.subplots(figsize=(5.5,4.6))
    ConfusionMatrixDisplay.from_predictions(yte,p,display_labels=['0: <2','1: >=2'],
                                             ax=ax,cmap='Blues',values_format='d')
    ax.set_title(f'Matriz de confusión: {nombre}')
    ax.set_xlabel('Clase predicha');ax.set_ylabel('Clase observada')
    guardar_fig(salida/f'confusion_{nombre.lower().replace(" ","_")}.png')
    return pipe,d,roc_curve(yte,proba)


def clasificacion(df,salida,n):
    d=muestrear_estratificado(df,n,'ATENCIONES_FRECUENTES_E3')
    X=d[FEATURES]
    y=d['ATENCIONES_FRECUENTES_E3']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=SEED)
    print(f'Clasificación: {len(d):,} filas; entrenamiento {len(Xtr):,}; prueba {len(Xte):,}.')
    # Ajuste k únicamente en conjunto de entrenamiento, nunca sobre prueba.
    Xt,Xv,yt,yv=train_test_split(Xtr,ytr,test_size=.2,stratify=ytr,random_state=SEED)
    filas_k=[]
    for k in [5,11,21]:
        mod=Pipeline([('preparar',preprocess()),('modelo',KNeighborsClassifier(n_neighbors=k,weights='distance',n_jobs=-1))])
        mod.fit(Xt,yt)
        filas_k.append({'k':k,'f1_validacion':f1_score(yv,mod.predict(Xv),zero_division=0)})
    pd.DataFrame(filas_k).to_csv(salida/'seleccion_k.csv',index=False)
    k=max(filas_k,key=lambda x:(x['f1_validacion'],-x['k']))['k']
    fig,ax=plt.subplots(figsize=(7.5,4.1))
    ax.plot([x['k'] for x in filas_k],[x['f1_validacion'] for x in filas_k],marker='o')
    ax.set(xlabel='Número de vecinos (k)',ylabel='F1 positivo, validación',title='Selección de k sin utilizar el conjunto de prueba')
    guardar_fig(salida/'seleccion_k.png')
    modelos={
        'Referencia':DummyClassifier(strategy='most_frequent'),
        'CART':DecisionTreeClassifier(criterion='gini',max_depth=6,min_samples_leaf=100,
                                      class_weight='balanced',random_state=SEED),
        # Árbol con entropía: referencia basada en ganancia de información.
        # No debe denominarse C4.5 completo porque scikit-learn no usa gain ratio.
        'Arbol_entropia':DecisionTreeClassifier(criterion='entropy',max_depth=6,min_samples_leaf=100,
                                      class_weight='balanced',random_state=SEED),
        'KNN':KNeighborsClassifier(n_neighbors=k,weights='distance',n_jobs=-1),
        'Logistica':LogisticRegression(max_iter=2000,class_weight='balanced',solver='lbfgs',random_state=SEED),
    }
    resumen=[]; rocs={}; fitted={}
    for nombre,modelo in modelos.items():
        p=Pipeline([('preparar',preprocess()),('modelo',modelo)])
        p,met,roc=evaluar(nombre,p,Xtr,ytr,Xte,yte,salida)
        fitted[nombre]=p;resumen.append(met);rocs[nombre]=roc
    tabla=pd.DataFrame(resumen)
    tabla.to_csv(salida/'metricas_clasificacion.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    tab=tabla.set_index('modelo')[['f1_1','recall_1','roc_auc']]
    tab.plot(kind='bar',ax=ax,rot=0)
    ax.set_ylim(0,1);ax.set_ylabel('Puntaje en prueba');ax.set_title('Comparación de modelos (conjunto de prueba independiente)')
    ax.legend(['F1 clase positiva','Recall clase positiva','ROC-AUC'])
    guardar_fig(salida/'comparacion_modelos.png')
    fig,ax=plt.subplots(figsize=(7,5.6))
    for nombre,(fpr,tpr,_) in rocs.items():
        valor=tabla.loc[tabla.modelo==nombre,'roc_auc'].iloc[0]
        ax.plot(fpr,tpr,label=f'{nombre} (AUC={valor:.3f})')
    ax.plot([0,1],[0,1],'k--',alpha=.4)
    ax.set(xlabel='Tasa de falsos positivos',ylabel='Sensibilidad',title='Curvas ROC en prueba')
    ax.legend()
    guardar_fig(salida/'curvas_roc.png')
    # Árbol CART (vista solo de los primeros tres niveles) y variables destacadas
    arb=fitted['CART'];feat=arb.named_steps['preparar'].get_feature_names_out()
    tree=arb.named_steps['modelo']
    fig,ax=plt.subplots(figsize=(17,9))
    plot_tree(tree,max_depth=3,feature_names=list(feat),class_names=['<2','>=2'],
              filled=True,rounded=True,fontsize=8,ax=ax)
    ax.set_title('CART (representación parcial: 3 primeros niveles)')
    guardar_fig(salida/'arbol_CART.png')
    importancias=pd.Series(tree.feature_importances_,index=feat).sort_values(ascending=False).head(12)
    fig,ax=plt.subplots(figsize=(9,5))
    importancias.sort_values().plot(kind='barh',ax=ax)
    ax.set(xlabel='Importancia Gini (asociación predictiva, no causalidad)',title='Variables destacadas del árbol CART')
    guardar_fig(salida/'importancia_CART.png')
    log=fitted['Logistica'];coef=pd.Series(log.named_steps['modelo'].coef_[0],
                index=log.named_steps['preparar'].get_feature_names_out())
    extremos=pd.concat([coef.nsmallest(6),coef.nlargest(6)]).sort_values()
    fig,ax=plt.subplots(figsize=(9,5))
    extremos.plot(kind='barh',ax=ax)
    ax.axvline(0,color='black',linewidth=.8)
    ax.set(xlabel='Coeficiente en log-odds',title='Regresión logística: coeficientes mayores en magnitud')
    guardar_fig(salida/'coeficientes_logisticos.png')
    pd.DataFrame({'variable':coef.index,'coeficiente':coef.values}).to_csv(salida/'coeficientes_logisticos.csv',index=False)
    return {'muestra':len(d),'entrenamiento':len(Xtr),'prueba':len(Xte),'k_elegido':int(k),
            'tasa_positivos':float(y.mean()),'metricas':resumen}


def clustering(df,salida,n):
    cols_num=[c for c in CLUSTER_NUM if c in df]
    cols_cat=[c for c in CLUSTER_CAT if c in df]
    df2=df.sample(n=min(n,len(df)),random_state=SEED).copy()
    # Costo y número de atenciones: escalas comprimidas por log1p; sin modificar la fuente E2.
    for c in cols_num:
        df2[c]=pd.to_numeric(df2[c],errors='coerce').clip(lower=0)
        if c in ['CANT_ATENCIONES','VALOR_NETO','DIAS_HOSP']:
            df2[c]=np.log1p(df2[c])
    prep=ColumnTransformer([
       ('num',Pipeline([('imputar',SimpleImputer(strategy='median')),
                        ('escalar',StandardScaler())]),cols_num),
       ('cat',Pipeline([('imputar',SimpleImputer(strategy='most_frequent')),
                        ('onehot',OneHotEncoder(handle_unknown='ignore'))]),cols_cat),
    ])
    X=prep.fit_transform(df2[cols_num+cols_cat])
    # Evaluación silhouette solo en submuestra fija y sin volver a entrenar en prueba.
    rng=np.random.default_rng(SEED)
    ix=rng.choice(X.shape[0],min(1800,X.shape[0]),replace=False)
    XX=X[ix]
    ks=range(2,7)
    rows=[]
    for k in ks:
        km=KMeans(n_clusters=k,random_state=SEED,n_init=10)
        labels=km.fit_predict(X)
        sc=silhouette_score(XX,km.predict(XX))
        rows.append({'k':k,'inercia':km.inertia_,'silueta_submuestra':sc})
    ev=pd.DataFrame(rows)
    ev.to_csv(salida/'seleccion_clusters.csv',index=False)
    # Mejor silueta como criterio explícito, no promesa de clusters clínicamente válidos.
    mejor=int(ev.sort_values(['silueta_submuestra','k'],ascending=[False,True]).iloc[0]['k'])
    modelo=KMeans(n_clusters=mejor,random_state=SEED,n_init=10)
    cluster=modelo.fit_predict(X)
    df2['CLUSTER_E3']=cluster
    for nombre,y,label in [('codo_inercia.png','inercia','Inercia (menor es mejor)'),
                           ('seleccion_silueta.png','silueta_submuestra','Coeficiente de silueta')]:
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        ax.plot(ev.k,ev[y],marker='o')
        ax.set(xlabel='Número de grupos (k)',ylabel=label,title=('Método del codo' if 'codo' in nombre else 'Selección por silueta'))
        guardar_fig(salida/nombre)
    # Reducción de dimensionalidad solo para visualización, K-means entrenó sobre TODAS las variables.
    coords=PCA(n_components=2,random_state=SEED).fit_transform(X.toarray() if hasattr(X,'toarray') else X)
    fig,ax=plt.subplots(figsize=(9,6))
    sc=ax.scatter(coords[:,0],coords[:,1],c=cluster,s=5,alpha=.45,cmap='tab10',rasterized=True)
    ax.set(xlabel='Componente principal 1',ylabel='Componente principal 2',
           title=f'Clusters K-Means proyectados a 2D mediante PCA (k={mejor})')
    fig.colorbar(sc,ax=ax,label='Grupo asignado')
    guardar_fig(salida/'clusters_PCA_2D.png')
    fig,ax=plt.subplots(figsize=(9,6))
    jitter=np.random.default_rng(SEED).normal(scale=.12,size=len(df2))
    size=df2['CANT_ATENCIONES'].apply(np.expm1).to_numpy()+jitter
    sc=ax.scatter(df2['EDAD'],size,c=cluster,s=7,alpha=.35,cmap='tab10',rasterized=True)
    ax.set(xlabel='Edad (años)',ylabel='Cantidad de atenciones (con dispersión gráfica)',
           title='Grupos en variables originales: edad y cantidad de atenciones')
    fig.colorbar(sc,ax=ax,label='Cluster')
    guardar_fig(salida/'clusters_edad_atenciones_2D.png')
    # Interpretación de datos expresados nuevamente en unidades originales.
    perfil=df2[['CLUSTER_E3','EDAD','CANT_ATENCIONES']+[c for c in ['VALOR_NETO','DIAS_HOSP'] if c in df2]].copy()
    for c in ['CANT_ATENCIONES','VALOR_NETO','DIAS_HOSP']:
        if c in perfil:perfil[c]=np.expm1(perfil[c])
    resumen=perfil.groupby('CLUSTER_E3').agg(['mean','median','count'])
    resumen.to_csv(salida/'perfiles_cluster_completos.csv')
    perfiles=perfil.groupby('CLUSTER_E3').agg(n=('EDAD','size'),edad_media=('EDAD','mean'),
          atenciones_media=('CANT_ATENCIONES','mean')).reset_index()
    if 'VALOR_NETO' in perfil:
        perfiles['valor_neto_medio']=perfil.groupby('CLUSTER_E3')['VALOR_NETO'].mean().values
    if 'DIAS_HOSP' in perfil:
        perfiles['dias_hosp_media']=perfil.groupby('CLUSTER_E3')['DIAS_HOSP'].mean().values
    perfiles.to_csv(salida/'perfiles_cluster.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.5,5))
    ancho=0.4;indices=np.arange(len(perfiles));
    ax.bar(indices-ancho/2,perfiles['edad_media'],ancho,label='Edad media (años)')
    ax2=ax.twinx()
    ax2.bar(indices+ancho/2,perfiles['atenciones_media'],ancho,color='tab:orange',label='Atenciones medias')
    ax.set_xticks(indices,perfiles.CLUSTER_E3);ax.set_xlabel('Grupo')
    ax.set_ylabel('Edad media (años)');ax2.set_ylabel('Atenciones medias')
    ax.set_title('Descripción de perfiles por cluster (escalas independientes)')
    guardar_fig(salida/'perfiles_clusters.png')
    # No exportar pacientes ni identificadores.
    return {'muestra':len(df2),'k_elegido':mejor,'silueta':float(ev.loc[ev.k==mejor,'silueta_submuestra'].iloc[0]),
            'perfiles':perfiles.to_dict('records')}


def reporte_texto(info,c,k,salida):
    met=pd.DataFrame(c['metricas']).set_index('modelo')
    partes=[
      'EVALUACIÓN 3 - RESULTADOS GENERADOS AUTOMÁTICAMENTE CON CSV REAL',
      'Importante: resultados descriptivos; no representan validación clínica.',
      '',f"Archivo original: {info['originales']:,} filas; muestra de base: {info['muestra_inicial']:,}; tras deduplicación: {info['tras_deduplicacion']:,}; válidas para E3: {info['tras_validacion']:,}.",
      'La semilla 42 reproduce ESTA ejecución, no garantiza la misma submuestra exacta de E2.',
      'El CSV filtrado puede tener un total distinto de 570.437. La E2 usó un pipeline distinto.',
      f"Etiqueta >=2 atenciones: positivos {info['objetivo_1']:,}; negativos {info['objetivo_0']:,}.",
      'Diagnósticos vacíos: '+('recodificados a NO (supuesto de E2, por validar).' if info['comparar_e2'] else 'categoría SIN_REGISTRO (sin asumir ausencia clínica).'),
      '',f"CLASIFICACIÓN - muestra {c['muestra']:,}, entrenamiento {c['entrenamiento']:,}, prueba {c['prueba']:,}; KNN con k={c['k_elegido']} seleccionado en validación.",
    ]
    for nombre,row in met.iterrows():
       partes.append(f'{nombre}: Accuracy={row.accuracy:.4f}; Accuracy balanceada={row.balanced_accuracy:.4f}; Precision={row.precision_1:.4f}; Recall={row.recall_1:.4f}; F1={row.f1_1:.4f}; ROC-AUC={row.roc_auc:.4f}.')
    partes.extend(['',f"CLUSTERING - muestra {k['muestra']:,}; K-Means k={k['k_elegido']}; silueta aproximada (submuestra)={k['silueta']:.4f}."])
    for perf in k['perfiles']:
        partes.append('Grupo {CLUSTER_E3}: n={n:.0f}; edad media={edad_media:.2f}; atenciones medias={atenciones_media:.2f}'.format(**perf))
    partes.extend(['','LIMITACIONES: no existe secuencia temporal verificable, imputación de diagnósticos incierta, muestras de algoritmos distintas y PCA utilizado solo como visualización.','No atribuir estos resultados al documento E2 sin cotejar los datos y la semilla.'])
    (salida/'resumen_para_informe.txt').write_text('\n'.join(partes),encoding='utf-8')


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--csv',required=True,type=Path,help='CSV original de la Evaluación 2')
    arg.add_argument('--salida',type=Path,default=Path('resultados_e3'))
    arg.add_argument('--n-base',type=int,default=650000)
    arg.add_argument('--n-clas',type=int,default=30000)
    arg.add_argument('--n-cluster',type=int,default=20000)
    arg.add_argument('--comparar-e2',action='store_true',help='Imputa diagnósticos faltantes a NO, igual supuesto que el informe anterior')
    args=arg.parse_args()
    if not args.csv.is_file():raise FileNotFoundError(f'No existe CSV: {args.csv}')
    args.salida.mkdir(parents=True,exist_ok=True)
    df,info=preparar(args.csv,args.n_base,args.comparar_e2)
    c=clasificacion(df,args.salida,args.n_clas)
    k=clustering(df,args.salida,args.n_cluster)
    reporte_texto(info,c,k,args.salida)
    (args.salida/'metadatos_ejecucion.json').write_text(json.dumps({'base':info,'clasificacion':c,'clustering':k},indent=2,ensure_ascii=False),encoding='utf-8')
    print('FIN. Resultados y figuras en:',args.salida.resolve())

if __name__=='__main__':main()
