"""Tests for the Orientation class.
"""

import numpy as np
import pytest

from walle.core import Orientation, UnitQuaternion, Quaternion


class TestOrientation(object):
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

    def test_init_empty(self):
        """Tests that the default constructor returns an identity quaternion.
        """
        ori = Orientation()
        actual = ori._quat
        expected = UnitQuaternion()
        assert actual == expected

    def test_init_rot_vec_valid_arr(self):
        """Tests orientation init with rotation vector ndarray.
        """
        rot_vec = self.rotation_vector(90)
        ori = Orientation(rot_vec)
        actual_axis, actual_theta = ori._quat.axis_angle
        expected_axis, expected_theta = np.array([0, 0, 1]), np.deg2rad(90)
        assert np.allclose(actual_axis, expected_axis) and np.isclose(actual_theta, expected_theta)

    def test_init_rot_vec_valid_list(self):
        """Tests orientation init with rotation vector list.
        """
        rot_vec = self.rotation_vector(90).tolist()
        ori = Orientation(rot_vec)
        actual_axis, actual_theta = ori._quat.axis_angle
        expected_axis, expected_theta = np.array([0, 0, 1]), np.deg2rad(90)
        assert np.allclose(actual_axis, expected_axis) and np.isclose(actual_theta, expected_theta)

    def test_init_rot_vec_invalid_list(self):
        """Tests orientation init with invalid rotation vector list.
        """
        rot_vec = self.rotation_vector(90).tolist()
        rot_vec = [rot_vec[0], rot_vec[1], [rot_vec[2]]]
        with pytest.raises(ValueError):
            Orientation(rot_vec)

    def test_init_rot_vec_invalid_arr(self):
        """Tests orientation init with invalid rotation vector ndarray.
        """
        rot_vec = np.random.randn(4)
        with pytest.raises(ValueError):
            Orientation(rot_vec)

    def test_init_axisang_valid_ndarray_float(self):
        """Test orientation with valid axis-angle (ndarray, float).
        """
        expected_axis, expected_theta = self.axis_angle_vector(90)
        ori = Orientation(expected_axis, expected_theta)
        actual_axis, actual_theta = ori._quat.axis_angle
        assert np.allclose(actual_axis, expected_axis) and np.isclose(actual_theta, expected_theta)

    def test_init_axisang_valid_list_float(self):
        """Test orientation with valid axis-angle (list, float).
        """
        expected_axis, expected_theta = self.axis_angle_vector(90)
        expected_axis = expected_axis.tolist()
        ori = Orientation(expected_axis, expected_theta)
        actual_axis, actual_theta = ori._quat.axis_angle
        assert np.allclose(actual_axis, expected_axis) and np.isclose(actual_theta, expected_theta)

    def test_init_axisang_invalid(self):
        """Test orientation with invalid axis-angle initialization.
        """
        with pytest.raises(ValueError):
            Orientation([1, 2, 3, 4], 0)

    def test_quaternion_rot_from_to_quat(self):
        """Tests the quaternion that rotates `from_quat` to `to_quat`.
        """
        quat_from = UnitQuaternion.random()
        quat_to = UnitQuaternion.random()
        rotation = Orientation.from_quats(quat_from, quat_to)
        actual = (rotation * quat_from).quat
        expected = quat_to
        assert actual == expected

    def test_quaternion_rot_from_to_vec_identity(self):
        """Tests the quaternion that rotates a vector to itself.

        Should return the indentity quaternion [1, 0, 0, 0].
        """
        x = np.array([1, 0, 0])
        quat = Orientation.from_vecs(x, x).quat
        assert quat.is_identity()

    def test_quaternion_rot_from_to_vec_random(self):
        """Tests the quaternion that rotates a vector to another.
        """
        x = np.random.randn(3)
        quat = UnitQuaternion.random()
        y = quat * x
        expected = y
        actual = Orientation.from_vecs(x, y) * x
        assert np.allclose(actual, expected)