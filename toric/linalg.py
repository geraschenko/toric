from typing import Tuple
import numpy as np

def exgcd(a: int, b: int) -> np.ndarray:
  """Extended GCD algorithm.

  Args:
    a: an integer.
    b: an integer.

  Returns:
    A 2x2 integer matrix M of determinant 1 so that M @ [a, b] = [gcd(a, b), 0].
    If a divides b, M[0, 1] is guaranteed to be 0.
  """
  # Ensure a and b are non-negative, so that // division works as expected.
  a_sign = -1 if a < 0 else 1
  a *= a_sign
  b_sign = -1 if b < 0 else 1
  b *= b_sign

  # We run Euclid's algorithm to compute gcd(a, b) with invertible row
  # operations on the column vector [a, b], and track these row operations by
  # augmenting by the identity.
  M = np.array([[a, 1, 0],
                [b, 0, 1]], dtype=object)
  M = M[::-1]  # swap the rows; required for the guarantee when a divides b.
  while M[1, 0] != 0:
    q = M[0, 0] // M[1, 0]
    M[0] -= q * M[1]
    M = M[::-1]  # swap the rows

  # The first column of M is now [gcd(a, b), 0], and the row operations we've
  # applied are given by left multiplication by M[:, 1:].
  g = M[0, 0]
  M = M[:, 1:]

  # Correct for the sign-adjustment we did at the beginning.
  M *= [a_sign, b_sign]

  # Fix the sign of the determinant, using M[0, 0] * a + M[0, 1] * b = g.
  if g != 0:
    M[1] = [- b_sign * b // g, a_sign * a // g]
  else:
    # a and b must both be 0, in which case M is the identity.
    pass

  return M


def inv_2x2_det1(M: np.ndarray) -> np.ndarray:
  """Matrix inverse of a 2x2 matrix with determinant 1."""
  assert M.shape == (2, 2) and (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0] == 1)
  return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]], dtype=object)


def normal_form(A: np.ndarray, return_inverses: bool=False
    ) -> Tuple[np.ndarray, ...]:
  """Sorta like Smith normal form, but without divisibility guarantees.

   https://en.wikipedia.org/wiki/Smith_normal_form.

  Args:
    A: an integer matrix.
    return_inverses: if True, inverses of S and T are returned.

  Returns:
    A tuple of integer matrices (S, D, T) (or (S, D, T, Sinv, Tinv) if
    return_inverses is True) satisfying
    * A == S @ D @ T
    * D is diagonal with the same shape as A
    * S and T are square of determinant 1, with inverses Sinv and Tinv.
    The inverses are included because np.linalg.inv only works for floats, so
    even though S and T have exact integer inverses, getting them with numpy
    alone is a pain.
  """
  # We "clear" a column by applying row operations which replace the its first
  # entry with the gcd of its entries and make all the other entries 0.  Then
  # we similarly clear the first row with column operations. This may mess up
  # the first column (though only if the first entry was replaced by a proper
  # divisor during row-clearing), so we keep alternating between clearing the
  # row and the column until they're both clear. Then we apply the same
  # procedure to clear the second row and second column, and so on.
  # As we apply row and column operations, we update S, T, Sinv, Tinv to
  # maintain the relations
  # S @ D @ T = A
  # Tinv @ T = I
  # S @ Sinv = I
  D = A.copy().astype(object)
  S, T = np.eye(D.shape[0], dtype=object), np.eye(D.shape[1], dtype=object)
  if return_inverses: Sinv, Tinv = S.copy(), T.copy()

  def clear_row(i):
    """Clears the i-th row of D, updating T and Tinv appropriately.

    Assumes rows and columns smaller than i are already clear.
    Returns False if the row was already clear.
    """
    if (D[i, i+1:] == 0).all():
      return False
    for j in range(i + 1, D.shape[1]):
      M = exgcd(D[i, i], D[i, j]).T
      D[:, [i, j]] = D[:, [i, j]] @ M
      T[[i, j]] = inv_2x2_det1(M) @ T[[i, j]]
      if return_inverses: Tinv[:, [i, j]] = Tinv[:, [i, j]] @ M
    assert (S @ D @ T == A).all()
    if return_inverses: assert (Tinv @ T == np.eye(len(T))).all()
    return True

  def clear_col(i):
    """Clears the i-th column of D, updating S and Sinv appropriately.

    Assumes rows and columns smaller than i are already clear.
    Returns False if the column was already clear.
    """
    if (D[i+1:, i] == 0).all():
      return False
    for j in range(i+1, D.shape[0]):
      M = exgcd(D[i, i], D[j, i])
      D[[i, j]] = M @ D[[i, j]]
      S[:, [i, j]] = S[:, [i, j]] @ inv_2x2_det1(M)
      if return_inverses: Sinv[[i, j]] = M @ Sinv[[i, j]]
    assert (S @ D @ T == A).all()
    if return_inverses: assert (S @ Sinv == np.eye(len(S))).all()
    return True

  for i in range(min(*D.shape)):
    clear_col(i)
    while True:
      if not clear_row(i):
        break
      if not clear_col(i):
        break

  if return_inverses:
    return S, D, T, Sinv, Tinv
  else:
    return S, D, T


def zero_pad(x: np.ndarray, length: int) -> np.ndarray:
  """x, padded with 0's so that it has length at least length."""
  assert x.ndim == 1
  return np.concatenate([x, np.zeros(max(0, length - len(x)), dtype=x.dtype)])


def kernel(A):
  """Returns a matrix whose columns span the null space of A."""
  S, D, T, Sinv, Tinv = normal_form(A, return_inverses=True)
  kernel_D = zero_pad(np.diag(D), len(T)) == 0
  return Tinv[:, kernel_D]


def cokernel(A):
  """Returns a matrix whose rows span the annihilator of the image of A."""
  S, D, T, Sinv, Tinv = normal_form(A, return_inverses=True)
  kernel_D = zero_pad(np.diag(D), len(S)) == 0
  return Sinv[kernel_D]


def relations(generators):
  """Returns the matrix of relations for a given set of generators."""
  return kernel(generators)


def index_in_saturation(A):
  """Returns the index of the image of A in its saturation."""
  S, D, T = normal_form(A)
  d = np.diag(D)
  return abs(np.prod(d[d != 0]))


def saturation(A):
  """Returns a matrix whose columns span the saturation of the image of A."""
  S, D, T = normal_form(A)
  kernel_D = zero_pad(np.diag(D), len(S)) == 0
  # TODO: this can be row reduced to make it prettier.
  return S[~kernel_D]
