"""An API for dealing with orientation and rotations in 3-D space.
"""

import numpy as np

from walle.core import constants, quaternion, utils
from walle.core.matrix import RotationMatrix
from walle.core.orthogonal import is_proper_rotm


class Orientation(object):
    """A convenience class for manipulating 3-D orientations and rotations.

    Attributes:
        rotm (ndarray): A 3x3 rotation matrix.
        axis_angle (tuple): axis angle.
        rot_vec (ndarray): rotation vector.
        quat (``UnitQuaternion``): unit quaternion.

    References:
        [1]: http://marc-b-reynolds.github.io/quaternions/2017/08/08/QuatRotMatrix.html
        [2]: https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        [3]: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
    """
    def __init__(self, *args):
        """Initializes the orientation in any of the following ways:

            u, theta: (array_like, float) Axis-angle. Represents a
                single rotation by a given angle `theta` in radians
                about a fixed axis represented by the unit vector `u`.
            rot_vec: (array_like) Rotation vector. Corresponds
                to the 3-element representation of the axis-angle,
                i.e. `rot_vec = theta * u`.
            rotm: (array_like) a `3x3` orthogonal rotation matrix.
            quat: (Quaternion or UnitQuaternion) a unit quaternion. The
                quaternion is normalized if it's a `Quaternion` object.

        If no arguments are passed, an identity Orientation is initialized.
        """
        if len(args) == 0:
            self._quat = quaternion.UnitQuaternion()
        elif len(args) == 1:
            if utils.is_iterable(args[0]):
                if len(args[0]) == 3 and utils.is_1d_iterable(args[0]):  # rot_vec
                    self._rot_vec = np.asarray(args[0], dtype="float64")
                    self._rotvec2quat()
                elif len(args[0]) == 3 and np.asarray(args[0]).shape == (3, 3):  # rotm
                    rotm = np.asarray(args[0], dtype="float64")
                    self._rotm2quat(rotm)
                else:
                    raise ValueError("[!] Could not parse constructor arguments.")
            elif isinstance(args[0], quaternion.Quaternion):  # quat
                self._quat = quaternion.UnitQuaternion(args[0])
            else:
                raise ValueError("[!] Expecting array_like or Quaternion.")
        elif len(args) == 2:
            if utils.is_iterable(args[0]) and isinstance(args[1], (int, float)):  # axang
                if len(args[0]) != 3:
                    raise ValueError("[!] Axis must be length 3 array_like.")
                self._u = np.asarray(args[0], dtype="float64")
                self._theta = float(args[1])
                if np.isclose(self._theta, 0.):  # no unique axis, default to i
                    self._u = np.array([1, 0, 0], dtype="float64")
                if not np.isclose(np.dot(self._u, self._u), 1.):
                    self._u /= np.linalg.norm(self._u)
                self._axang2quat()
            else:
                raise ValueError("[!] Expecting (axis, angle) tuple.")
        else:
            raise ValueError("[!] Incorrect number of arguments.")

    def __repr__(self):
        s = "{}[θ: {:.5f}, u: ({:.5f}, {:.5f}, {:.5f})]"
        u, theta = self._quat.axis_angle
        return s.format(self.__class__.__name__, theta, *list(u))

    def __eq__(self, other):
        raise NotImplementedError("[!] Use `eq_rot` or `eq_ori` for equality checks.")

    def __mul__(self, other):
        """Orientation multiplication.
        """
        if utils.is_iterable(other):
            if utils.is_batch_vectors(other) or utils.is_single_vector(other):
                return np.dot(self.rotm, np.asarray(other, dtype="float64"))
            else:
                if all(isinstance(x, Orientation) for x in other):  # batch of orientations
                    accessor = lambda x: x._quat
                elif all(isinstance(x, quaternion.UnitQuaternion) for x in other):  # batch of quaternions
                    accessor = lambda x: x
                else:
                    raise NotImplemented
                quat_prod = self._quat
                for o in other:
                    quat_prod = quat_prod * accessor(o)
                    quat_prod.fnormalize(True)
                return self.__class__(quat_prod)
        else:
            if isinstance(other, Orientation):
                other = other._quat
            elif isinstance(other, quaternion.UnitQuaternion):
                other = other
            else:
                raise NotImplemented
            quat_prod = self._quat * other
            quat_prod.fnormalize(True)
            return self.__class__(quat_prod)

    def eq_rot(self, other):
        """This checks whether two orientations correspond to the same rotation.
        """
        if isinstance(other, Orientation):
            if self._quat.dot(other._quat) > 1 - constants.EPS:
                return True
            return False
        else:
            raise NotImplementedError("[!] Other must be an instance of {}.".format(self.__class__.__name__))

    def eq_ori(self, other):
        """This checks whether two orientations correspond to the same orientation.
        """
        if isinstance(other, Orientation):
            if abs(self._quat.dot(other._quat)) > 1 - constants.EPS:
                return True
            return False
        else:
            raise NotImplementedError("[!] Other must be an instance of {}.".format(self.__class__.__name__))

    def inv(self, inplace=False):
        """Returns the inverse of this orientation.
        """
        if not inplace:
            return self.__class__(self._quat.inv())
        self._quat.inv(True)

    def _rotvec2quat(self):
        """Converts a rotation vector to a unit-quaternion.
        """
        self._theta = np.linalg.norm(self._rot_vec)
        if np.isclose(self._theta, 0.):
            self._u = np.array([1, 0, 0], dtype="float64")
            self._theta = 0.
        else:
            self._u = self._rot_vec / self._theta
        self._axang2quat()

    def _axang2quat(self):
        """Converts an axis-angle to a unit-quaternion.
        """
        s = np.cos(self._theta / 2)
        v = self._u * np.sin(self._theta / 2)
        self._quat = quaternion.UnitQuaternion(s, v)

    def _rotm2quat(self, rotm):
        """Converts a rotation matrix to a unit-quaternion.
        """
        m00 = rotm[0, 0]
        m01 = rotm[0, 1]
        m02 = rotm[0, 2]
        m10 = rotm[1, 0]
        m11 = rotm[1, 1]
        m12 = rotm[1, 2]
        m20 = rotm[2, 0]
        m21 = rotm[2, 1]
        m22 = rotm[2, 2]
        if is_proper_rotm(rotm):  # Shepperd's algorithm
            if m22 >= 0:
                a = m00 + m11
                b = m10 - m01
                c = 1. + m22
                if a >= 0:
                    s = c + a
                    q = [s, m21 - m12, m02 - m20, b]
                else:
                    s = c - a
                    q = [b, m02 + m20, m21 + m12, s]
            else:
                a = m00 - m11
                b = m10 + m01
                c = 1. - m22
                if a >= 0:
                    s = c + a
                    q = [m21 - m12, s, b, m02 + m20]
                else:
                    s = c - a
                    q = [m02 - m20, b, s, m21 + m12]
            q = 0.5 * np.array(q) * (1. / np.sqrt(s))
        else:  # Bar-Itzhack's algorithm
            Q = np.array([
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ])
            Q /= 3.
            eig_values, eig_vectors = np.linalg.eigh(Q)
            q = eig_vectors[np.argmax(eig_values)][[3, 0, 1, 2]]
        if q[0] < 0.:
            q = -q
        self._quat = quaternion.UnitQuaternion(q)

    @classmethod
    def from_quats(cls, u, v):
        """Computes the rotation that rotates quaternion `u` to quaternion `v`.

        Args:
            u: (UnitQuaternion) the starting point quaternion.
            v: (UnitQuaternion) the end point quaternion.
        """
        if all(isinstance(x, quaternion.UnitQuaternion) for x in [u, v]):
            return cls(v * u.inv())
        else:
            raise ValueError("[!] Inputs must all be unit-quaternions.")

    @classmethod
    def from_vecs(cls, u, v):
        """Computes the rotation that rotates vector `u` to vector `v`.

        Implements the algorithm described in [3].

        Args:
            u: (array_like) the starting 3-D vector.
            v: (array_like) the final 3-D vector.
        """
        norm_uv = np.sqrt(np.dot(u, u) * np.dot(v, v))
        s = norm_uv + np.dot(u, v)
        if s < (1e-6 * norm_uv):  # if u and v point in opposite directions
            # rotate 180 degrees about any orthogonal axis
            s = 0.
            if abs(u[0]) > abs(u[2]):
                v = np.array([-u[1], u[0], 0.])
            else:
                v = np.array([0., -u[2], u[1]])
        else:
            v = np.cross(u, v)
        return cls(quaternion.UnitQuaternion(s, v))

    @classmethod
    def randquat(cls):
        """Generates an orientation by randomly sampling a quaternion.
        """
        return cls(quaternion.UnitQuaternion.random())

    @classmethod
    def randrotm(cls):
        """Generates an orientation by randomly sampling a rotation matrix.
        """
        return cls(RotationMatrix.random())

    @property
    def rotm(self):
        return self._quat.rotm

    @property
    def quat(self):
        return self._quat

    @property
    def axis_angle(self):
        return self._quat.axis_angle

    @property
    def rot_vec(self):
        return self._quat.rot_vec