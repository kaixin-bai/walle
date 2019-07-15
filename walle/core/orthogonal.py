"""Orthogonlization methods for rotation matrices.
"""

from abc import ABC, abstractmethod

import numpy as np

from walle.core.utils import is_iterable


class BaseOrthogonalizer(ABC):
    """Base class for orthogonalizing near-rotation matrices.
    """
    def __init__(self):
        pass

    def _check_input(self, arr):
        """Checks that the input is a `3x3` array_like.
        """
        if is_iterable(arr):
            if np.asarray(arr).shape == (3, 3):
                return True
            return False
        return False

    @staticmethod
    def is_proper_rotm(rotm):
        """Checks that the input is a proper orthogonal `3×3` matrix.
        """
        is_orth = np.allclose(np.dot(rotm, rotm.T), np.eye(3))
        is_proper = np.isclose(np.linalg.det(rotm), 1.)
        return is_orth and is_proper

    @abstractmethod
    def _renormalize(self, rotm):
        """Re-orthogonalizes an input matrix.
        """
        pass

    def renormalize(self, rotm):
        if not self._check_input(rotm):
            raise ValueError("[!] Must be a 3x3 array_like.")
        else:
            rotm = np.asarray(rotm, dtype="float64")
        if BaseOrthogonalizer.is_proper_rotm(rotm):
            print("[!] Rotation matrix is already proper.")
            return rotm
        self._renormalize(rotm)


class SVDOrthogonalizer(BaseOrthogonalizer):
    """Implements the algorithm defined in [1].

    [1]: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    [2]: https://www.cse.iitb.ac.in/~ajitvr/CS763_Spring2017/procrustes.pdf
    """
    def _renormalize(self, rotm):
        u, _, vt = np.linalg.svd(rotm)
        rot = u @ vt
        if np.linalg.det(rot) < 0.:  # in case we have a reflection matrix
            print("[!] Special reflection case...")
            vt[2, :] *= -1  # negate the row of v corresponding to the smallest eigenvalue
            rot = u @ vt
        return rot


class DCMOrthogonalizer(BaseOrthogonalizer):
    """Implements the algorithm defined in [1].

    The algorithm works in 3 steps:
        1. Rotates the `X` and `Y` axes of a rotation
            matrix using the error factor.
        2. Constructs the `Z` axis using the corrected
            `X` and `Y` axes (cross-product).
        3. Normalizes the rows to unit-norm.

    [1]: https://wiki.paparazziuav.org/w/images/e/e5/DCMDraft2.pdf
    """
    def _renormalize(self, rotm):
        half_error = 0.5 * np.dot(rotm[0, :], rotm[1, :])
        rotm[0, :] -= (half_error * rotm[1, :])
        rotm[1, :] -= (half_error * rotm[0, :])
        rotm[2, :] = np.cross(rotm[0, :], rotm[1, :])
        scale = 0.5 * (3 - np.sum(rotm * rotm, axis=1))
        return scale * rotm


class QROrthogonalizer(BaseOrthogonalizer):
    """QR factorization.
    """
    def _renormalize(self, rotm):
        return np.linalg.qr(rotm)[0]


class NearestOrthogonalizer(BaseOrthogonalizer):
    """Implements the algorithm defined in [1].

    This may not always return a proper rotation matrix.

    [1]: http://people.csail.mit.edu/bkph/articles/Nearest_Orthonormal_Matrix.pdf
    """
    def _renormalize(self, rotm):
        pass


class GramSchmidtOrthogonalizer(BaseOrthogonalizer):
    """Gram-Schmidt orthogonalization.
    """
    def _renormalize(self, rotm):
        raise ValueError("[!] Avoid this at all costs.")


# aliases
is_proper_rotm = BaseOrthogonalizer.is_proper_rotm