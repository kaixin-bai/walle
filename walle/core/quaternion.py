"""Quaternions and Unit-Quaternions.
"""

import numpy as np

from walle.core import constants, utils


class Quaternion(object):
    """A quaternion class for performing quaternion algebra.

    Attributes:
        scalar (:obj:`float`): The real component of the quaternion.
        vector (:obj:`ndarray`): The complex component of the quaternion.
        array (:obj:`ndarray`): An array containing the real component
            followed by the complex component.
        pybullet (:obj:`ndarray`): An array containing the complex component
            followed by the real component.
    """
    def __init__(self, *args):
        """Initializes the quaternion in any of the following ways:

        Args:
            s (:obj:`float`): The real component. Assumes complex component is zero.
            v (``array_like``): The complex component. Assumes real component is zero.
            q (``array_like``): An iterable of 4 floats defining the real and complex components.
                You can also provide them separately as two function arguments ``s, v``.
            quat (:obj:`Quaternion`): An instance of this class. This is useful for copying
                a quaternion or creating a ``UnitQuaternion`` object from an existing quaternion.

        If no arguments are passed, a zero quaternion is initialized.
        """
        self._q = np.empty(4, dtype="float64")
        if len(args) == 0:
            self._q = np.zeros(4, dtype="float64")
        elif len(args) == 1:
            if utils.is_iterable(args[0]):
                if len(args[0]) == 4:  # [s, x, y, z]
                    self._q[0] = float(args[0][0])
                    self._q[1:] = args[0][1:]
                elif len(args[0]) == 3:  # pure quaternion
                    self._q[0] = 0.
                    self._q[1:] = args[0]
                else:
                    raise ValueError("[!] Expecting a length 4 array_like or pure quaternion.")
            elif isinstance(args[0], Quaternion):
                self._q = np.array(args[0]._q, dtype="float64")
            else:  # real quaternion
                self._q[0] = float(args[0])
                self._q[1:] = np.zeros(3, dtype="float64")
        elif len(args) == 2:  # s, v
            if utils.is_iterable(args[0]):
                raise ValueError("[!] `s` must be a float.")
            self._q[0] = float(args[0])
            if not utils.is_iterable(args[1]):
                raise ValueError("[!] `v` must be array_like.")
            self._q[1:] = args[1]
        else:
            raise ValueError("[!] Incorrect number of arguments.")

    def __repr__(self):
        s = "{}[s: {:.5f}, v: ({:.5f}, {:.5f}, {:.5f})]"
        return s.format(self.__class__.__name__, self._q[0], *list(self._q[1:]))

    def __eq__(self, other):
        """Checks whether two quaternions are equal.

        This function checks that two quaternions are
        component-wise equal, i.e. that their scalar
        and vector parts are element-wise equal within
        a tolerance.

        If this function evaluates to True, it does
        not imply that the two quaternions represent
        the same orientation or orientation. In fact,
        ``q`` and ``-q`` represent the same orientation
        but not the same rotation, yet the result of
        this function would yield False.
        """
        return np.allclose(self._q, other._q)

    def __ne__(self, other):
        """Checks whether two quaternions are not equal.
        """
        return not (self == other)

    def __neg__(self):
        """Negates the quaternion.
        """
        return self.__class__(-self._q)

    def __abs__(self):
        """Returns the absolute value of the quaternion.

        Note:
            This is equivalent to the norm of the quaternion.
        """
        return self.norm()  # abs(x) = norm(x)

    def __add__(self, other):
        """Left quaternion addition.

        Addition can be performed with:

        - A ``Quaternion`` or ``quaternion_like`` object.
        - An ``array_like`` of length 3 to represent a purely-complex quaternion.
        - A :obj:`float` to represent a purely-real quaternion.
        """
        if isinstance(other, Quaternion):
            return self.__class__(self._q + other._q)
        elif utils.is_iterable(other):
            if len(other) == 3:
                return self.__class__(
                    self._q[0],
                    self._q[1:] + other,
                )
            elif len(other) == 4:
                return self.__class__(self._q + other)
            else:
                raise ValueError("[!] Expecting length 3 or length 4 array_like.")
        elif isinstance(other, (int, float)):
            return self.__class__(
                self._q[0] + float(other),
                self._q[1:],
            )
        else:
            raise NotImplemented

    def __radd__(self, other):
        """Right quaternion addition.
        """
        return self + other  # quaternion addition is commutative

    def __iadd__(self, other):
        """In-place quaternion addition.
        """
        return self + other

    def __sub__(self, other):
        """Left quaternion subtraction.

        Subtraction can be performed with:

        - A ``Quaternion`` or ``quaternion_like`` object.
        - An ``array_like`` of length 3 to represent a purely-complex quaternion.
        - A :obj:`float` to represent a purely-real quaternion.
        """
        if isinstance(other, Quaternion):
            return self.__class__(self._q - other._q)
        elif utils.is_iterable(other):
            if len(other) == 3:
                return self.__class__(
                    self._q[0],
                    self._q[1:] - other,
                )
            elif len(other) == 4:
                return self.__class__(self._q - other)
            else:
                raise ValueError("[!] Expecting length 3 or length 4 array_like.")
        elif isinstance(other, (int, float)):
            return self.__class__(
                self._q[0] - float(other),
                self._q[1:],
            )
        else:
            raise NotImplemented

    def __rsub__(self, other):
        """Right quaternion subtraction.
        """
        return -self + other

    def __isub__(self, other):
        """In-place subtraction.
        """
        return self - other

    def __mul__(self, other):
        """Quaternion multiplication.

        Args:
            other (``quaternion_like`` or :obj:`float`): The quaternion to
                multiply with or scalar to scale by.

        Returns:
            ``Quaternion``

        Interpreted as quaternion-quaternion or scalar-quaternion multiplication.
        """
        if isinstance(other, Quaternion):
            return self.__class__(
                (self._q[0] * other._q[0]) - np.dot(self._q[1:], other._q[1:]),
                (self._q[0] * other._q[1:]) + (other._q[0] * self._q[1:]) + np.cross(self._q[1:], other._q[1:]),
            )
        elif utils.is_iterable(other):
            other = np.asarray(other, dtype="float64")
            if len(other) == 3:
                return self.__class__(
                    - np.dot(self._q[1:], other),
                    (self._q[0] * other) + np.cross(self._q[1:], other),
                )
            elif len(other) == 4:
                return self.__class__(
                    (self._q[0] * other[0]) - np.dot(self._q[1:], other[1:]),
                    (self._q[0] * other[1:]) + (other[0] * self._q[1:]) + np.cross(self._q[1:], other[1:]),
                )
            else:
                raise ValueError("[!] Expecting length 3 or length 4 array_like.")
        elif isinstance(other, (int, float)):  # scalar multiplication
            return self.__class__(
                float(other) * self._q[0],
                float(other) * self._q[1:],
            )
        else:
            raise NotImplemented

    def __rmul__(self, other):
        """Quaternion multiplication.
        """
        if isinstance(other, Quaternion):
            return self.__class__(
                (other._q[0] * self._q[0]) - np.dot(other._q[1:], self._q[1:]),
                (other._q[1:] * self._q[0]) + (self._q[1:] * other._q[0]) + np.cross(other._q[1:], self._q[1:]),
            )
        elif utils.is_iterable(other):
            other = np.asarray(other, dtype="float64")
            if len(other) == 3:
                return self.__class__(
                    - np.dot(other, self._q[1:]),
                    (other * self._q[0]) + np.cross(other, self._q[1:]),
                )
            elif len(other) == 4:
                return self.__class__(
                    (other[0] * self._q[0]) - np.dot(other[1:], self._q[1:]),
                    (other[1:] * self._q[0]) + (self._q[1:] * other[0]) + np.cross(other[1:], self._q[1:]),
                )
            else:
                raise ValueError("[!] Expecting length 3 or length 4 array_like.")
        elif isinstance(other, (int, float)):  # scalar multiplication
            return self.__class__(
                float(other) * self._q[0],
                float(other) * self._q[1:],
            )
        else:
            raise NotImplemented

    def __imul__(self, other):
        """Performs in-place multiplication.
        """
        return self * other

    def __pow__(self, x):
        """Raises the quaternion to a real exponent `x`.

        Args:
            x (:obj:`float`): The exponent to raise by.

        Returns:
            ``Quaternion``
        """
        if not isinstance(x, (int, float)):
            raise ValueError("[!] Exponent must be real.")
        n, theta = self.polar()
        length = (self.norm() ** x)
        s = length * np.cos(x * theta)
        v = (np.sin(x * theta) * length) * n
        return self.__class__(s, v)

    def conjugate(self, inplace=False):
        """Returns the complex conjugate of the quaternion.

        Args:
            inplace (bool): If ``True``, performs the operation in-place.
        """
        if inplace:
            self._q[1:] = - self._q[1:]
        else:
            return self.__class__(self._q[0], -self._q[1:])

    def norm(self):
        """Returns the L2 norm of the quaternion.

        Returns:
            :obj:`float`
        """
        return np.linalg.norm(self._q)

    def norm_squared(self):
        """Returns the square of the L2 norm of the quaternion.

        Returns:
            :obj:`float`
        """
        return np.dot(self._q, self._q)

    def inv(self, inplace=False):
        """Returns the inverse of the quaternion.

        Args:
            inplace (bool): If ``True``, performs the operation in-place.
        """
        norm_sq = self.norm_squared()
        if norm_sq < constants.EPS:
            raise ZeroDivisionError("[!] Norm is very close to 0.")
        if inplace:
            self.conjugate(True)
            self._q *= (1. / norm_sq)
        else:
            return (1. / norm_sq) * self.conjugate()

    def normalize(self, inplace=False):
        """Returns the normalized quaternion.

        Args:
            inplace (bool): If ``True``, performs the operation in-place.
        """
        if self.is_unit_norm():
            if inplace:
                return
            return self.__class__(self._q)
        norm = self.norm()
        if norm < constants.EPS:
            raise ZeroDivisionError("[!] Norm is very close to 0.")
        if inplace:
            self._q /= norm
            self._q[0] = self._q[0]
            self._q[1:] = self._q[1:]
        else:
            return self.__class__(self._q / norm)

    def fnormalize(self, inplace=False):
        """Fast normalization using first-order approximation [1]_.

        This should only be used if normalizing frequently when chaining rotations.

        Args:
            inplace (bool): If ``True``, performs the operation in-place.

        References:
            .. [1] Normalization with Padé Approximation,
                https://stackoverflow.com/a/12934750/4875916.
        """
        norm_sq = self.norm_squared()
        if abs(1 - norm_sq) < 2.107342e-08:
            if inplace:
                self._q *= (2 / (1 + norm_sq))
            else:
                return self.__class__(self._q * (2 / (1 + norm_sq)))
        else:
            if inplace:
                self._q *= (1 / np.sqrt(norm_sq))
            else:
                return self.__class__(self._q * (1 / np.sqrt(norm_sq)))

    def dot(self, other):
        """Performs the dot product of two quaternions.

        Args:
            other (``Quaternion``): Another quaternion.

        Returns:
            :obj:`float`
        """
        if isinstance(other, Quaternion):
            return np.dot(self._q, other._q)
        else:
            raise NotImplemented

    def angle(self, other):
        """Returns the angle between two quaternions.

        Args:
            other (``Quaternion``): Another quaternion.

        Returns:
            :obj:`float`: The angle in radians.
        """
        if isinstance(other, Quaternion):
            norm_a, norm_b = self.norm(), other.norm()
            if any(n < constants.EPS for n in [norm_a, norm_b]):
                raise ZeroDivisionError("[!] Norm is very close to 0.")
            return np.arccos(self.dot(other) / (norm_a * norm_b))
        else:
            raise NotImplemented

    def dist(self, other):
        """Returns the distance between two quaternions.

        Args:
            other (``Quaternion``): Another quaternion.

        Returns:
            :obj:`float`
        """
        if isinstance(other, Quaternion):
            return (self - other).norm()  # the norm of the difference
        else:
            raise NotImplemented

    def exp(self):
        """Returns the exponential of the quaternion.

        Returns:
            ``Quaternion``
        """
        norm_v = np.linalg.norm(self._q[1:])
        if norm_v < constants.EPS:
            raise ZeroDivisionError("[!] Norm is very close to 0.")
        exp_s = np.exp(self._q[0])
        q_exp = np.hstack([
            np.cos(norm_v),
            (self._v / norm_v) * np.sin(norm_v)
        ])
        return self.__class__(exp_s * q_exp)

    def log(self):
        """Returns the natural logarithm of the quaternion.

        Returns:
            ``Quaternion``
        """
        norm_q = self.norm()
        norm_v = np.linalg.norm(self._q[1:])
        if any(n < constants.EPS for n in [norm_v, norm_q]):
            raise ZeroDivisionError("[!] Norm is very close to 0.")
        q_log = np.hstack([
            np.log(norm_q),
            (self._q[1:] / norm_v) * np.arccos(self._q[0] / norm_q)
        ])
        return self.__class__(q_log)

    def polar(self):
        """Returns the polar form of the quaternion.

        Returns:
            ``Quaternion``s
        """
        norm_v = np.linalg.norm(self._q[1:])
        norm_q = self.norm()
        if any(n < constants.EPS for n in [norm_v, norm_q]):
            raise ZeroDivisionError("[!] Norm is very close to 0.")
        n = self._q[1:] / norm_v
        theta = np.arccos(self._q[0] / norm_q)
        return n, theta

    def is_unit_norm(self):
        """Checks whether the quaternion has unit-norm.
        """
        return np.isclose(self.norm_squared(), 1.)

    def is_pure(self):
        """Checks whether the quaternion is purely imaginary.
        """
        return np.isclose(self._q[0], 0.)

    def is_real(self):
        """Checks whether the quaternion is purely real.
        """
        return np.allclose(self._q[1:], 0.)

    def is_identity(self):
        """Checks whether the quaternion is the identity.
        """
        eye = np.array([1, 0, 0, 0], dtype="float64")
        return np.allclose(self._q, eye)

    @property
    def scalar(self):
        return self._q[0]

    @property
    def vector(self):
        return self._q[1:]

    @property
    def array(self):
        return self._q

    @property
    def pybullet(self):
        q = list(self._q)
        return q[1:4] + q[0:1]


class UnitQuaternion(Quaternion):
    """A unit-quaternion class for performing 3-D rotations.

    This class inherits from ``Quaternion`` and implements more
    efficient routines for rotations, inverse rotations and
    compositions of rotations.

    Attributes:
        axis_angle (:obj:`tuple`): The axis-angle representation of
            the unit quaternion ``(theta, axis)``.
        rot_vec (:obj:`ndarray`): The rotation vector representation of
            the unit quaternion ``theta*axis``.
        rotm (:obj:`ndarray`): The 3x3 rotation matrix representation of
            the unit quaternion.
    """
    def __init__(self, *args):
        """Initializes the unit quaternion.

        If no arguments are passed, an identity quaternion (no rotation) is initialized.
        """
        if len(args) == 0:
            super().__init__(1.)
        else:
            super().__init__(*args)

        # inplace normalization
        self.normalize(True)

        self._u = None
        self._theta = None
        self._rotm = None
        self._rot_vec = None

    def __add__(self, other):
        raise NotImplementedError("[!] Addition is not implemented.")

    def __sub__(self, other):
        raise NotImplementedError("[!] Subtraction is not implemented..")

    def __mul__(self, other):
        """Unit-quaternion multiplication.

        Multiplication is interpreted as a:

        - Rotation if ``other`` is a length 3 array_like.
        - Rotation if ``other`` is a pure quaternion.
        - Regular quaternion multiplication if ``other`` is a non-pure quaternion.

        Note:
            Quaternion-scalar multiplication is not supported
            because it has no effect on the unit-quaternion. This
            is because we normalize to unit-norm in the constructor.

            We use Rodrigues' formula rather than quaternion conjugation
            ``q * p * q.inv()`` since it is more efficient.

        Returns:
            An :obj:`ndarray`, ``Quaternion`` or ``UnitQuaternion``:
                - :obj:`ndarray`: The rotated vector if the multiplication was interpreted as a rotation.
                - ``Quaternion``: The quaternion product if the multiplication was interpreted as regular quaternion multiplication.
                - ``UnitQuaternion``: If both operands were unit-quaternions.
        """
        if utils.is_iterable(other):
            other = np.asarray(other, dtype="float64")
            if len(other) == 3:  # vector rotation
                return self._rodrigues_rotation(other)
            elif len(other) == 4:
                if other[0] == 0:  # pure quaternion, i.e. vector rotation
                    return self._rodrigues_rotation(other[1:])
                return super().__mul__(other)  # regular quaternion multiplication
            elif other.ndim == 2 and other.shape[0] > 1 and other.shape[1] == 3:  # batch of vectors
                res = np.empty_like(other)
                for i in range(other.shape[0]):
                    res[i] = self._rodrigues_rotation(other[i])
                return res
            else:
                raise ValueError("[!] array_like is not compatible.")
        elif isinstance(other, UnitQuaternion):
            return super().__mul__(other)  # the product of unit-quaternions is a unit-quaternion
        elif isinstance(other, Quaternion):
            if other.is_pure():  # pure quaternion, i.e. vector rotation
                return Quaternion(self._rodrigues_rotation(other._q[1:]))
            # regular quaternion multiplication
            res = super().__mul__(other)
            return Quaternion(res.array)
        else:
            raise NotImplemented

    def _rodrigues_rotation(self, other):
        """Rotates a vector using Rodrigues' formula.
        """
        a = np.cross(self._q[1:], other) + self._q[0] * other
        b = np.cross(self._q[1:], a)
        return b + b + other

    def _quat2axang(self):
        """Converts a quaternion to an axis-angle.
        """
        unit_i = np.array([1, 0, 0], dtype="float64")
        theta = 2 * np.arctan2(np.linalg.norm(self.vector), self.scalar)
        if theta < constants.EPS:  # no unique axis -> default to null axis
            self._u, self._theta = unit_i, 0.
        else:
            u = self.vector / np.sin(theta / 2)
            if not np.isclose(np.dot(u, u), 1.):
                u /= np.linalg.norm(u)
            self._u, self._theta = u, theta
        self._rot_vec = self._theta * self._u

    def _quat2rotm(self):
        """Converts a quaternion to an orthogonal rotation matrix.
        """
        a, b, c, d = self.array
        a_sq, b_sq, c_sq, d_sq = a**2, b**2, c**2, d**2
        two_ab = 2 * a * b
        two_bc = 2 * b * c
        two_ad = 2 * a * d
        two_bd = 2 * b * d
        two_ac = 2 * a * c
        two_cd = 2 * c * d
        self._rotm = np.array([
            [a_sq + b_sq - c_sq - d_sq, two_bc - two_ad, two_bd + two_ac],
            [two_bc + two_ad, a_sq - b_sq + c_sq - d_sq, two_cd - two_ab],
            [two_bd - two_ac, two_cd + two_ab, a_sq - b_sq - c_sq + d_sq],
        ])
        self._rotm[np.isclose(self._rotm, 0.)] = 0.  # zero-out very small values

    @classmethod
    def random(cls):
        """Generates a uniform random unit-quaternion as described in [2]_.

        Returns:
            ``UnitQuaternion``

        References:
            .. [2] Generating a Random Element of SO3,
                http://planning.cs.uiuc.edu/node198.html.
        """
        u0, u1, u2 = np.random.uniform(0, 1, 3)
        hs = np.empty(4, dtype="float64")
        two_pi_u1 = 2 * np.pi * u1
        two_pi_u2 = 2 * np.pi * u2
        s1 = np.sqrt(1 - u0)
        s2 = np.sqrt(u0)
        hs[0] = s1 * np.sin(two_pi_u1)
        hs[1] = s1 * np.cos(two_pi_u1)
        hs[2] = s2 * np.sin(two_pi_u2)
        hs[3] = s2 * np.cos(two_pi_u2)
        return cls(hs)

    def inv(self, inplace=False):
        """The inverse of a unit-quaternion is just its conjugate.

        Args:
            inplace (bool): If ``True``, performs the operation in-place.
        """
        if not inplace:
            return self.conjugate()
        self.conjugate(True)

    def angle(self, other):
        """Returns the angle (in radians) between two unit-quaternions.

        Args:
            other (``Quaternion`` or ``UnitQuaternion``): A unit-quaternion
                or quaternion in which case it is normalized to unit-norm.
        """
        if isinstance(other, Quaternion):
            other = other.normalize()  # enforce unit-quaternion
            norm_add = np.linalg.norm(self._q + other._q)
            norm_sub = np.linalg.norm(self._q - other._q)
            return 2 * np.arctan2(norm_sub, norm_add)
        else:
            raise NotImplemented

    def geodesic_norm(self, other):
        """Returns the geodesic distance between two unit-quaternions.

        Args:
            other (``Quaternion`` or ``UnitQuaternion``): A unit-quaternion
                or quaternion in which case it is normalized to unit-norm.
        """
        if isinstance(other, Quaternion):
            other = other.normalize()  # enforce unit-quaternion
            return np.arccos(2 * self.dot(other)**2 - 1)
        else:
            raise NotImplemented

    def slerp(self, other, t):
        """Spherical linear interpolation between two unit-quaternions.

        Args:
            other (`quaternion_like`): The quaternion to interpolate with.
            t (:obj:`float`): The interpolation parameter between 0 and 1.
        """
        if not isinstance(other, Quaternion):
            other = Quaternion(other)
        other = other.normalize()  # a valid rotation must have unit-norm
        if self.dot(other) < 0.:  # ensure shortest path
            other = -other  # reverse quaternion
        if utils.is_iterable(t):
            t = np.clip(t, 0, 1)
            res = []
            for t_ in t:
                res.append((other * self.inv())**t_ * self)
            return np.asarray(res)
        elif isinstance(t, (int, float)):
            return (other * self.inv())**float(t) * self
        else:
            raise ValueError("[!] `t` must be a float or array of floats.")

    @property
    def axis_angle(self):
        if self._rot_vec is None:
            self._quat2axang()
        return self._u, self._theta

    @property
    def rot_vec(self):
        if self._rot_vec is None:
            self._quat2axang()
        return self._rot_vec

    @property
    def rotm(self):
        if self._rotm is None:
            self._quat2rotm()
        return self._rotm