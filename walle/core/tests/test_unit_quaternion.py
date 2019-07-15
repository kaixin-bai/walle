"""Tests for the UnitQuaternion class.
"""

import numpy as np
import pytest

from walle.core import Quaternion, UnitQuaternion


class TestUnitQuaternion(object):
    def axis_angle_vector(self, deg):
        theta = np.deg2rad(deg)
        unit_vec = np.array([0, 0, 1])
        return unit_vec, theta

    def rotation_vector(self, deg):
        unit_vec, theta = self.axis_angle_vector(deg)
        return theta * unit_vec

    def quat_from_axang(self, deg):
        unit_vec, theta = self.axis_angle_vector(deg)
        s = np.cos(theta / 2)
        v = unit_vec * np.sin(theta / 2)
        return UnitQuaternion(s, v)

    def test_inv(self):
        """Tests the inverse of the unit quaternion.
        """
        quat = self.quat_from_axang(90)
        expected = quat.conjugate()
        actual = quat.inv()
        assert actual == expected

    def test_rotation_equiv(self):
        """Tests that Rodrigues' formula is equivalent to conjugation.
        """
        unit_quat = self.quat_from_axang(90)
        quat = Quaternion(unit_quat.array)
        pos_vec = np.array([1, 2, 3])
        actual = Quaternion(unit_quat * pos_vec)
        expected = quat * pos_vec * quat.inv()
        assert actual == expected

    def test_rotation_i2j(self):
        """Tests the rotation of unit vector `i` by 90 degrees.
        """
        quat = self.quat_from_axang(90)
        unit_i = np.array([1, 0, 0])
        actual = (quat * unit_i)
        expected = np.array([0, 1, 0])
        assert np.allclose(actual, expected)

    def test_rotation_i2negi(self):
        """Tests the rotation of unit vector `i` by 180 degrees.
        """
        quat = self.quat_from_axang(180)
        unit_i = np.array([1, 0, 0])
        actual = (quat * unit_i)
        expected = np.array([-1, 0, 0])
        assert np.allclose(actual, expected)

    def test_slerp_zero(self):
        """Tests slerp with `t=0`.
        """
        quat1 = self.quat_from_axang(90)
        quat2 = self.quat_from_axang(180)
        expected = quat1
        actual = quat1.slerp(quat2, 0)
        assert actual == expected

    def test_slerp_one(self):
        """Tests slerp with `t=1`.
        """
        quat1 = self.quat_from_axang(90)
        quat2 = self.quat_from_axang(180)
        expected = quat2
        actual = quat1.slerp(quat2, 1.)
        assert actual == expected

    def test_slerp_rand(self):
        """Tests slerp with a random `t`.

        Ref: https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
        """
        quat1 = self.quat_from_axang(90)
        quat2 = self.quat_from_axang(180)
        t = np.random.random()
        omega = quat1.angle(quat2)
        coeff1 = np.sin((1 - t) * omega) / np.sin(omega)
        coeff2 = np.sin(t * omega) / np.sin(omega)
        expected = UnitQuaternion((coeff1 * quat1.array) + (coeff2 * quat2.array))
        actual = quat1.slerp(quat2, t)
        assert actual == expected

    def test_rotation_n_times(self):
        """Tests the rotation of unit vector `i` by 90 degrees 4 times.

        We should essentially recover `i`.
        """
        quat = self.quat_from_axang(90)
        unit_i = np.array([1, 0, 0])
        actual = (quat**4 * unit_i)
        expected = unit_i
        assert np.allclose(actual, expected)

    def test_quaternion_to_axang(self):
        """Tests quaternion to axis-angle conversion.
        """
        quat = self.quat_from_axang(90)
        axis_actual, angle_actual = quat.axis_angle
        axis_expected, angle_expected = self.axis_angle_vector(90)
        assert np.isclose(angle_actual, angle_expected) and np.allclose(axis_actual, axis_expected)

    def test_quaternion_to_rot_vec(self):
        """Tests quaternion to rotation vector conversion.
        """
        quat = self.quat_from_axang(90)
        expected = self.rotation_vector(90)
        actual = quat.rot_vec
        assert np.allclose(actual, expected)