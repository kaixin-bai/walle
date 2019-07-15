"""An API that combines position and orientation in 3-D space.
"""

import numpy as np

from walle.core.orientation import Orientation
from walle.core.matrix import RotationMatrix
from walle.core import utils, quaternion


class Pose(object):
    """A convenience class for manipulating 3-D position and orientation.
    """
    def __init__(self, *args):
        """Initializes the 6-DOF pose in any of the following ways:

            pose: (array_like) A list of 6 floats. The first three
                represent the 3-D position [x, y, z] and the last
                three represent the 3-D orientation in rotation
                vector format [rx, ry, rz].
            transform: (array_like) A 4x4 transform representing
                the 3-D position and 3-D orientation.
            position, orientation: (array_like, `Orientation`) The
                first argument is an array_like of length 3 representing
                the 3-D position [x, y, z]. The second argument is an
                instance of the `Orientation` class representing the
                3-D orientation.
            position, rot_vec: (array_like, array_like) The first
                argument is an array_like of length 3 representing
                the 3-D position [x, y, z]. The second argument
                is an array_like of length 3 specifying the 3-D
                orientation in rotation vector format [rx, ry, rz].
            position, rotm: (array_like, array_like) The first
                argument is an array_like of length 3 representing
                the 3-D position [x, y, z]. The second argument
                is an array_like representing the 3-D orientation
                in rotation matrix format.
            position, quat: (array_like, UnitQuaternion) The first
                argument is an array_like of length 3 representing
                the 3-D position [x, y, z]. The second argument
                is an array_like representing the 3-D orientation
                in unit-quaternion format.

        If no arguments are passed, an identity Pose is initialized.
        """
        if len(args) == 0:
            self._position = np.zeros(3, dtype="float64")
            self._orientation = Orientation()
        elif len(args) == 1:  # pose
            if utils.is_1d_iterable(args[0]):
                if len(args[0]) == 6 or np.asarray(args[0]).shape == (1, 6):
                    pose = np.asarray(args[0]).astype("float64").squeeze()
                    self._position = pose[:3]
                    self._orientation = Orientation(pose[3:])
                else:
                    raise ValueError("[!] Expecting [x, y, z, rx, ry, rz] pose.")
            elif np.asarray(args[0]).shape == (4, 4):  # transform
                transform = np.asarray(args[0], dtype="float64")
                self._position = transform[:3, 3]
                self._orientation = Orientation(transform[:3, :3])
            else:
                raise ValueError("[!] Expecting 6-D pose or 4x4 transform.")
        elif len(args) == 2:
            if utils.is_iterable(args[0]) and len(args[0]) == 3:
                self._position = np.asarray(args[0], dtype="float64")
                if isinstance(args[1], Orientation):
                    self._orientation = Orientation(args[1].quat)
                elif isinstance(args[1], quaternion.UnitQuaternion):
                    self._orientation = Orientation(args[1])
                elif utils.is_iterable(args[1]):
                    if utils.is_1d_iterable(args[1]) and len(args[1]) == 3:
                        self._orientation = Orientation(np.asarray(args[1], dtype="float64"))
                    elif np.asarray(args[1]).shape == (3, 3):
                        self._orientation = Orientation(np.asarray(args[1], dtype="float64"))
                    else:
                        raise ValueError("[!] Expecting rotation vector or rotation matrix.")
                else:
                    raise ValueError("[!] Could not parse orientation argument.")
            else:
                raise ValueError("[!] Could not parse position argument.")
        else:
            raise ValueError("[!] Incorrect number of arguments.")

        # construct 4x4 transform
        self._transform = np.eye(4)
        self._transform[:3, :3] = self._orientation.rotm
        self._transform[:3, 3] = self._position

    def __repr__(self):
        s = "{}[Position: ({:.5f}, {:.5f}, {:.5f}), {}]"
        return s.format(self.__class__.__name__, *list(self._position), self._orientation)

    def __eq__(self, other):
        if isinstance(other, Pose):
            return np.allclose(self._transform, other._transform)
        elif isinstance(other, np.ndarray) and other.shape == (4, 4):
            return np.allclose(self._transform, other)
        else:
            raise NotImplementedError("[!] Other must be an instance of {}.".format(self.__class__.__name__))

    def __mul__(self, other):
        """Pose multiplication.
        """
        if isinstance(other, Pose):
            return self.__class__(self._transform @ other._transform)
        else:
            raise NotImplementedError("[!] Other must be an instance of {}.".format(self.__class__.__name__))

    def inv(self, inplace=False):
        """Returns the inverse of this pose.
        """
        if not inplace:
            return self.__class__(np.linalg.inv(self._transform))
        self._transform = np.linalg.inv(self._transform)
        self._position = self._transform[:3, 3]
        self._orientation.inv(True)

    def apply_rotz(self, angle, inplace=False):
        """Apply a rotation about the z-axis on the pose.
        """
        rotm = RotationMatrix.rotz(np.radians(angle))
        rotate_by = Pose([0, 0, 0], rotm)
        if inplace:
            rotated = rotate_by * self
            self._transform = rotated.transform
            self._position = self._transform[:3, 3]
            self._orientation = Orientation(self._transform[:3, :3])
        else:
            return (rotate_by * self)

    @property
    def transform(self):
        """Returns the 4x4 transform representation of the pose.
        """
        return self._transform

    @property
    def pose(self):
        """Returns the (position, rot_vec) representation of the pose.
        """
        return np.hstack([self._position, self._orientation.rot_vec])

    @property
    def position(self):
        """Returns the position part of the transformation.
        """
        return self._position

    @property
    def orientation(self):
        """Returns the orientation part of the transformation.
        """
        return self._orientation

    @property
    def rotm(self):
        """Returns the orientation as a rotation matrix.
        """
        return self._orientation.rotm

    @property
    def quat(self):
        """Returns the orientation as a unit-quaternion.
        """
        return self._orientation.quat

    @property
    def axis_angle(self):
        """Returns the orientation as an axis-angle.
        """
        return self._orientation.axis_angle

    @property
    def rot_vec(self):
        """Returns the orientation as a rotation vector.
        """
        return self._orientation.rot_vec
