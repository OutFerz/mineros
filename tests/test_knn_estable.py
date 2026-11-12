import unittest
import numpy as np
from scipy.sparse import csr_matrix
from knn_estable import KNNDeterminista


class TestKNNDeterminista(unittest.TestCase):
    def test_empate_distancia_se_resuelve_por_id_no_por_clase(self):
        X=np.array([[-1.], [1.]])
        m=KNNDeterminista(n_neighbors=1).fit(X,[0,1],sample_ids=[20,10])
        self.assertEqual(m.predict([[0.]])[0],1)
        # Cambiar clases no cambia la selección del vecino.
        m.fit(X,[1,0],sample_ids=[20,10])
        self.assertEqual(m.predict([[0.]])[0],0)

    def test_invariante_orden_y_tamano_lote(self):
        X=np.array([[0.],[0.],[1.],[-1.],[2.]])
        y=np.array([0,1,1,0,0]); ids=np.array([40,10,30,20,50])
        a=KNNDeterminista(3,batch_size=1).fit(X,y,ids)
        p=np.array([4,2,0,3,1])
        b=KNNDeterminista(3,batch_size=100).fit(csr_matrix(X[p]),y[p],ids[p])
        np.testing.assert_array_equal(a.predict_proba([[0.],[.5],[2.]]), b.predict_proba([[0.],[.5],[2.]]))

    def test_ceros_excluyen_vecinos_no_cero(self):
        m=KNNDeterminista(3).fit([[0.],[1.],[2.]],[1,0,0])
        np.testing.assert_array_equal(m.predict_proba([[0.]]),[[0.,1.]])

    def test_pesos_inversos_y_empate_voto(self):
        m=KNNDeterminista(2).fit([[0.],[3.]],[0,1])
        np.testing.assert_allclose(m.predict_proba([[1.]]), [[2/3,1/3]])
        self.assertEqual(m.predict([[1.5]])[0],0)

    def test_ids_duplicados_rechazados(self):
        with self.assertRaises(ValueError):
            KNNDeterminista(1).fit([[0.],[1.]],[0,1],sample_ids=[3,3])
