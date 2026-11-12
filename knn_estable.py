"""K-NN euclidiano ponderado con desempate explícito e independiente de la clase."""
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class KNNDeterminista(ClassifierMixin, BaseEstimator):
    """Ordena por distancia² redondeada a 12 decimales y luego por ID de fila.

    Los ID proceden de la muestra antes del train/test split. No se utiliza la
    etiqueta para resolver empates. Se mantienen exactamente k vecinos y pesos
    inversos a la distancia; si hay vecinos a distancia cero, solo estos votan.
    """
    def __init__(self, n_neighbors=5, batch_size=128, distance_decimals=12):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.distance_decimals = distance_decimals

    @staticmethod
    def _dense(X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError('Se requiere una matriz finita de dos dimensiones.')
        return X

    def fit(self, X, y, sample_ids=None):
        X = self._dense(X)
        y = np.asarray(y)
        if len(y) != len(X) or y.ndim != 1:
            raise ValueError('X e y deben tener el mismo número de filas.')
        if not isinstance(self.n_neighbors, (int, np.integer)) or not 1 <= self.n_neighbors <= len(X):
            raise ValueError('n_neighbors debe estar entre 1 y el número de filas.')
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1:
            raise ValueError('batch_size debe ser un entero positivo.')
        ids = np.arange(len(X)) if sample_ids is None else np.asarray(sample_ids)
        if ids.ndim != 1 or len(ids) != len(X) or len(np.unique(ids)) != len(ids):
            raise ValueError('sample_ids debe contener un ID único por fila.')
        order = np.argsort(ids, kind='stable')
        self.X_ = X[order]
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.y_ = encoded[order]
        self.sample_ids_ = ids[order]
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        X = self._dense(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Número de variables incompatible con entrenamiento.')
        result = np.zeros((len(X), len(self.classes_)))
        for start in range(0, len(X), self.batch_size):
            # cdist calcula las diferencias directamente, sin cancelación por X²+Y²-2XY.
            d2 = np.round(cdist(X[start:start+self.batch_size], self.X_, metric='sqeuclidean'), self.distance_decimals)
            indices = np.argsort(d2, axis=1, kind='stable')[:, :self.n_neighbors]
            distances = np.sqrt(np.take_along_axis(d2, indices, axis=1))
            zero = distances == 0
            with np.errstate(divide='ignore'):
                weights = np.where(zero, 0., 1. / distances)
            rows_zero = zero.any(axis=1)
            weights[rows_zero] = zero[rows_zero].astype(float)
            labels = self.y_[indices]
            for cls in range(len(self.classes_)):
                result[start:start+len(indices), cls] = (weights * (labels == cls)).sum(axis=1)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        # En empate de voto se elige la primera clase en orden; regla explícita.
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
