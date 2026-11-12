"""Árbol de tipo C4.5 para clasificación binaria con datos mixtos.

Implementa gain ratio, pruebas multirrama para categorías, umbrales continuos
y poda pesimista basada en Wilson. No es el ejecutable histórico de Quinlan
ni implementa su conversión opcional del árbol a un conjunto de reglas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class _Nodo:
    n: int
    positivos: int
    atributo: str | None = None
    umbral: float | None = None
    hijos: dict = field(default_factory=dict)


class C45Classifier(ClassifierMixin, BaseEstimator):
    """Árbol de decisión C4.5, objetivo binario 0/1.

    Las categorías no registradas se conservan como SIN_REGISTRO.
    Si aparece una categoría desconocida al predecir, se usa la
    distribución de la clase del nodo alcanzado.
    """

    def __init__(self, max_depth=6, min_samples_leaf=100,
                 confidence_factor=0.25, min_gain=1e-8, prune=True,
                 threshold=0.5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.confidence_factor = confidence_factor
        self.min_gain = min_gain
        self.prune = prune
        self.threshold = threshold

    @staticmethod
    def _entropia(p, n):
        n = np.asarray(n, dtype=float)
        p = np.asarray(p, dtype=float)
        q = np.divide(p, n, out=np.zeros_like(p), where=n > 0)
        return -(np.where((q > 0) & (q < 1), q * np.log2(np.clip(q, 1e-15, 1)), 0)
                 + np.where((q > 0) & (q < 1),
                            (1-q) * np.log2(np.clip(1-q, 1e-15, 1)), 0))

    @staticmethod
    def _entropia_ramas(tamanos):
        frecuencias = tamanos / np.sum(tamanos)
        return -np.sum(frecuencias * np.log2(frecuencias))

    def _mejor_corte(self, X, y, ids, col, es_num):
        valores = X[col][ids]
        n = len(ids)
        padre = float(self._entropia(y[ids].sum(), n))
        if es_num:
            orden = np.argsort(valores, kind="mergesort")
            vv = valores[orden]
            yy = y[ids][orden]
            cortes = np.flatnonzero(vv[:-1] != vv[1:])
            cortes = cortes[(cortes + 1 >= self.min_samples_leaf)
                            & (n - cortes - 1 >= self.min_samples_leaf)]
            if not len(cortes):
                return None
            n1 = (cortes + 1).astype(float)
            n2 = n - n1
            p1 = np.cumsum(yy)[cortes]
            p2 = yy.sum() - p1
            ganancias = (padre - (n1/n) * self._entropia(p1, n1)
                         - (n2/n) * self._entropia(p2, n2))
            splitinfo = self._entropia(n1, n)
            ratios = np.divide(ganancias, splitinfo,
                               out=np.zeros_like(ganancias), where=splitinfo > 0)
            # C4.5 selecciona umbrales por gain ratio, pero descarta
            # cortes con una ganancia inferior a la media de cortes útiles.
            # Maximizar solo ganancia absoluta no representa ese criterio.
            validas = ganancias > self.min_gain
            if not np.any(validas):
                return None
            ganancia_min = max(self.min_gain, float(np.mean(ganancias[validas])))
            elegibles = validas & (ganancias >= ganancia_min - 1e-12)
            puntajes = np.where(elegibles, ratios, -np.inf)
            ix = int(np.argmax(puntajes))
            umbral = float((vv[cortes[ix]] + vv[cortes[ix]+1]) / 2)
            mascara = valores <= umbral
            return {"feature": col, "threshold": umbral,
                    "gain": float(ganancias[ix]), "ratio": float(ratios[ix]),
                    "groups": {"<=": ids[mascara], ">": ids[~mascara]}}
        cats, inv = np.unique(valores, return_inverse=True)
        if len(cats) <= 1:
            return None
        tamanos = np.bincount(inv).astype(float)
        if np.any(tamanos < self.min_samples_leaf):
            return None
        positivos = np.bincount(inv, weights=y[ids], minlength=len(cats))
        ganancia = padre - float(np.sum((tamanos/n) *
                                       self._entropia(positivos, tamanos)))
        splitinfo = self._entropia_ramas(tamanos)
        if ganancia <= self.min_gain or splitinfo <= 0:
            return None
        return {"feature": col, "threshold": None, "gain": ganancia,
                "ratio": ganancia/splitinfo,
                "groups": {str(cat): ids[inv == i] for i, cat in enumerate(cats)}}

    def _construir(self, X, y, ids, disponibles, profundidad):
        positivos = int(y[ids].sum())
        nodo = _Nodo(n=len(ids), positivos=positivos)
        if (profundidad >= self.max_depth or positivos in (0, len(ids))
                or len(ids) < 2*self.min_samples_leaf):
            return nodo
        candidatos = []
        for col in disponibles:
            candidato = self._mejor_corte(
                X, y, ids, col, col in self.numeric_features_)
            if candidato is not None:
                candidatos.append(candidato)
        if not candidatos:
            return nodo
        # C4.5 evita premiar una gran razón de ganancia con ganancia insignificante.
        ganancia_media = np.mean([c["gain"] for c in candidatos])
        posibles = [c for c in candidatos if c["gain"] >= ganancia_media-1e-12]
        mejor = max(posibles, key=lambda c: (c["ratio"], c["gain"]))
        nodo.atributo = mejor["feature"]
        nodo.umbral = mejor["threshold"]
        disponibles_hijos = [
            c for c in disponibles if c != nodo.atributo or nodo.umbral is not None]
        for categoria, child_ids in mejor["groups"].items():
            nodo.hijos[categoria] = self._construir(
                X, y, child_ids, disponibles_hijos, profundidad+1)
        return nodo

    def _costo_hoja(self, nodo):
        errores = min(nodo.positivos, nodo.n-nodo.positivos)
        z = self._z
        tasa = errores/nodo.n
        div = 1 + z*z/nodo.n
        limite = (tasa + z*z/(2*nodo.n) + z*np.sqrt(
            tasa*(1-tasa)/nodo.n + z*z/(4*nodo.n**2))) / div
        return nodo.n * limite

    def _podar(self, nodo):
        if not nodo.hijos:
            return self._costo_hoja(nodo)
        costo_subarbol = sum(self._podar(h) for h in nodo.hijos.values())
        costo_hoja = self._costo_hoja(nodo)
        if costo_hoja <= costo_subarbol + 0.1:
            nodo.atributo = None
            nodo.umbral = None
            nodo.hijos = {}
            return costo_hoja
        return costo_subarbol

    @staticmethod
    def _contar_nodos(nodo):
        if not nodo.hijos:
            return 1, 1
        conteos = [C45Classifier._contar_nodos(h) for h in nodo.hijos.values()]
        return 1 + sum(a for a, _ in conteos), sum(b for _, b in conteos)

    def fit(self, X, y):
        if not 0 < self.confidence_factor < 0.5:
            raise ValueError("confidence_factor debe estar entre 0 y 0.5")
        if not 0 < self.threshold < 1:
            raise ValueError('threshold debe estar entre 0 y 1')
        if self.min_samples_leaf < 1 or self.max_depth < 1:
            raise ValueError("max_depth y min_samples_leaf deben ser positivos")
        data = pd.DataFrame(X).copy()
        target = np.asarray(y, dtype=int)
        if (len(data) != len(target) or len(np.unique(target)) != 2
                or not set(np.unique(target)) <= {0, 1}):
            raise ValueError("Se necesita clasificación binaria 0/1 con ambas clases")
        self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        self.n_features_in_ = data.shape[1]
        self.numeric_features_ = [
            c for c in data if pd.api.types.is_numeric_dtype(data[c])]
        self.categorical_features_ = [
            c for c in data if c not in self.numeric_features_]
        valores = {}
        self.fill_num_ = {}
        for col in data.columns:
            if col in self.numeric_features_:
                serie = pd.to_numeric(data[col], errors="coerce").to_numpy(
                    dtype=float)
                mediana = np.nanmedian(serie)
                if not np.isfinite(mediana):
                    raise ValueError(f"La columna {col} no contiene números válidos")
                valores[col] = np.where(np.isfinite(serie), serie, mediana)
                self.fill_num_[col] = float(mediana)
            else:
                valores[col] = data[col].fillna(
                    "SIN_REGISTRO").astype(str).to_numpy()
        self._z = NormalDist().inv_cdf(1-self.confidence_factor)
        self.classes_ = np.asarray([0, 1])
        self.root_ = self._construir(
            valores, target, np.arange(len(target)), list(data.columns), 0)
        self.unpruned_nodes_, self.unpruned_leaves_ = self._contar_nodos(self.root_)
        self.root_feature_before_pruning_ = self.root_.atributo
        if self.prune:
            self._podar(self.root_)
        self.nodes_, self.leaves_ = self._contar_nodos(self.root_)
        self.pruned_nodes_ = self.unpruned_nodes_ - self.nodes_
        self.decision_threshold_ = float(self.threshold)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "root_")
        data = pd.DataFrame(X).copy()
        if list(data.columns) != list(self.feature_names_in_):
            raise ValueError("Columnas y orden distintos del entrenamiento")
        for col in self.numeric_features_:
            numeros = pd.to_numeric(data[col], errors="coerce").to_numpy(
                dtype=float)
            data[col] = np.where(
                np.isfinite(numeros), numeros, self.fill_num_[col])
        for col in self.categorical_features_:
            data[col] = data[col].fillna("SIN_REGISTRO").astype(str)
        resultado = np.empty((len(data), 2), dtype=float)
        for i, valores in enumerate(data.itertuples(index=False, name=None)):
            fila = dict(zip(self.feature_names_in_, valores))
            nodo = self.root_
            while nodo.hijos:
                clave = (("<=" if fila[nodo.atributo] <= nodo.umbral else ">")
                         if nodo.umbral is not None
                         else str(fila[nodo.atributo]))
                if clave not in nodo.hijos:
                    break
                nodo = nodo.hijos[clave]
            positivo = (nodo.positivos+1)/(nodo.n+2)
            resultado[i] = [1-positivo, positivo]
        return resultado

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold_).astype(int)

    def resumen_ramas(self, niveles=2, ramas_por_nodo=4):
        """Resumen textual legible para una evidencia en imagen."""
        check_is_fitted(self, "root_")
        lineas = []
        def visitar(nodo, etiqueta, nivel):
            prefijo = "  " * nivel
            pronostico = int(nodo.positivos >= nodo.n/2)
            linea = f"{prefijo}{etiqueta}: n={nodo.n:,}; positivos={nodo.positivos:,}; clase={pronostico}"
            if nodo.hijos:
                linea += f"; divide por {nodo.atributo}"
            lineas.append(linea)
            if nivel >= niveles:
                return
            for clave, hijo in list(nodo.hijos.items())[:ramas_por_nodo]:
                criterio = f"{nodo.atributo} {clave}"
                if nodo.umbral is not None:
                    criterio += f" {nodo.umbral:.2f}"
                visitar(hijo, criterio, nivel+1)
            if len(nodo.hijos) > ramas_por_nodo:
                lineas.append(f"{prefijo}  ... {len(nodo.hijos)-ramas_por_nodo} ramas adicionales")
        visitar(self.root_, "RAÍZ", 0)
        return "\n".join(lineas)
