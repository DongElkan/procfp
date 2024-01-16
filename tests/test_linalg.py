import unittest

import numpy as np

from procfp.core.linalg import cal_ldl, solve_x


class Testlinalg(unittest.TestCase):
    def test_ldl(self):
        t = np.random.randn(10, 4)
        a = np.dot(t.T, t)
        ml = np.linalg.cholesky(a)
        d = np.diag(ml) ** 2
        for i in range(4):
            ml[i][i] = 1.

        ml_c, d_c = cal_ldl(a.astype(np.float64))

        ml_c2 = np.asarray(ml_c)
        a_c = np.dot(np.dot(ml_c2, np.diag(d_c)), ml_c2.T)

        self.assertTrue(np.allclose(a, a_c))
        self.assertTrue(np.allclose(d, d_c))

    def test_solve_linear(self):
        t = np.random.randn(10, 4)
        a = np.dot(t.T, t).astype(np.float64)
        z = np.random.randn(4).astype(np.float64)
        x, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
        x_c = solve_x(a, z)
        print(x, np.asarray(x_c))
        print(z, np.dot(a, x_c))
        self.assertTrue(np.allclose(x, x_c))


if __name__ == "__main__":
    unittest.main()
