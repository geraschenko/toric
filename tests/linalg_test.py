from absl.testing import absltest, parameterized
from toric import linalg
import numpy as np
import math

class TestLinAlg(parameterized.TestCase):
  def test_normal_form(self):
    # TODO(geraschenko): set a seed.
    A = np.random.randint(-5, 5, size=(5, 5))
    S, D, T, Sinv, Tinv = linalg.normal_form(A, return_inverses=True)
    np.testing.assert_array_equal(S @ D @ T, A)
    np.testing.assert_array_equal(S @ Sinv, np.eye(len(S)))
    np.testing.assert_array_equal(T @ Tinv, np.eye(len(T)))

  @parameterized.parameters([(1, 1), (-1, 1), (12, 20)])
  def test_exgcd(self, a, b):
    M = linalg.exgcd(a, b)
    g = math.gcd(a, b)
    np.testing.assert_array_equal(M @ [a, b], [g, 0])
    if b % a == 0:
      self.assertEqual(M[0, 1], 0)



if __name__ == '__main__':
  absltest.main()
