import unittest
import numpy as np
import pandas as pd
from c45_arbol import C45Classifier


class TestC45(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 1200
        self.x = pd.DataFrame({
            "EDAD": rng.integers(18, 90, n),
            "DEPARTAMENTO": rng.choice(["A", "B", "C"], n),
            "SEXO": rng.choice(["F", "M"], n),
            "CON_DX_OBESIDAD": rng.choice(["SI", "NO", "SIN_REGISTRO"], n),
        })
        self.y = (((self.x.EDAD > 60) & (self.x.CON_DX_OBESIDAD == "SI")) |
                  (self.x.DEPARTAMENTO == "C")).astype(int)

    def test_entrena_predice_y_poda(self):
        m = C45Classifier(max_depth=6, min_samples_leaf=20).fit(self.x, self.y)
        p = m.predict_proba(self.x)
        self.assertEqual(p.shape, (len(self.x), 2))
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
        self.assertGreater(np.mean(m.predict(self.x) == self.y), .90)
        self.assertTrue(m.resumen_ramas())

    def test_categoria_desconocida_y_nulos(self):
        m = C45Classifier(max_depth=5, min_samples_leaf=20).fit(self.x, self.y)
        x = self.x.iloc[:2].copy()
        x.loc[x.index[0], "DEPARTAMENTO"] = "NUEVO"
        x.loc[x.index[1], "CON_DX_OBESIDAD"] = None
        p = m.predict_proba(x)
        self.assertEqual(p.shape, (2, 2))
        self.assertTrue(np.isfinite(p).all())

    def test_diagnostico_poda_y_umbral(self):
        m = C45Classifier(max_depth=6, min_samples_leaf=20, prune=False,
                          threshold=0.20).fit(self.x, self.y)
        self.assertGreater(m.nodes_, 1)
        self.assertEqual(m.unpruned_nodes_, m.nodes_)
        self.assertEqual(m.pruned_nodes_, 0)
        self.assertIsNotNone(m.root_feature_before_pruning_)
        p = m.predict_proba(self.x)[:, 1]
        np.testing.assert_array_equal(m.predict(self.x), (p >= 0.20).astype(int))
        podado = C45Classifier(max_depth=6, min_samples_leaf=20,
                               prune=True).fit(self.x, self.y)
        self.assertLessEqual(podado.nodes_, podado.unpruned_nodes_)
        self.assertGreaterEqual(podado.pruned_nodes_, 0)

    def test_gain_ratio_numerico_no_simple_ganancia(self):
        x = pd.DataFrame({"EDAD": [1, 2, 3, 4, 5, 6, 7, 8]*25})
        y = np.asarray(([0, 0, 0, 0, 1, 1, 1, 1])*25)
        m = C45Classifier(min_samples_leaf=10, prune=False).fit(x,y)
        self.assertEqual(m.root_.atributo, "EDAD")
        self.assertAlmostEqual(m.root_.umbral, 4.5)

    def test_resumen_respeta_umbral_y_no_mayoria(self):
        # Predominan ceros, pero el umbral 0.20 clasifica el nodo como 1.
        x = pd.DataFrame({"EDAD": [50] * 10})
        y = np.array([0] * 7 + [1] * 3)
        m = C45Classifier(max_depth=2, min_samples_leaf=2,
                          threshold=0.20, prune=False).fit(x,y)
        resumen = m.resumen_ramas()
        self.assertIn("pred_nodo=1", resumen)
        self.assertIn("p1=0.333", resumen)
        self.assertTrue(np.all(m.predict(x) == 1))

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            C45Classifier(confidence_factor=1).fit(self.x, self.y)
        with self.assertRaises(ValueError):
            C45Classifier(min_samples_leaf=0).fit(self.x, self.y)

    def test_corte_numerico(self):
        x = pd.DataFrame({"EDAD": [20]*100 + [65]*100, "SEXO": ["F", "M"]*100})
        y = np.array([0]*100 + [1]*100)
        m = C45Classifier(max_depth=3, min_samples_leaf=10).fit(x, y)
        self.assertEqual(m.root_.atributo, "EDAD")
        self.assertAlmostEqual(np.mean(m.predict(x) == y), 1.0)


if __name__ == "__main__":
    unittest.main()
