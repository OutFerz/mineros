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
