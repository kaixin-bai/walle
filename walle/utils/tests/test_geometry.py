"""Tests geometry methods.
"""

import pytest

import numpy as np

from walle.utils import geometry
from walle.core import RotationMatrix, Pose


def test_estimate_rigid_transform_rotm_rotation():
    xs = np.random.randn(5, 3)
    rotm_expected = RotationMatrix.rotz(np.radians(np.random.uniform(0, 360)))
    ys = (rotm_expected @ xs.T).T
    rotm_actual, _ = geometry.estimate_rigid_transform_rotm(xs, ys)
    assert np.allclose(rotm_actual, rotm_expected)


def test_estimate_rigid_transform_rotm_identity():
    xs = np.random.randn(5, 3)
    rotm_expected = np.eye(3)
    ys = xs.copy()
    rotm_actual, _ = geometry.estimate_rigid_transform_rotm(xs, ys)
    assert np.allclose(rotm_actual, rotm_expected)


def test_estimate_rigid_transform_rotm_rotation_and_translation():
    xs = np.random.randn(5, 3)
    rotm = RotationMatrix.rotz(np.radians(np.random.uniform(0, 360)))
    tvec = 5 * np.random.randn(1, 3)
    transform_expected = Pose.transform_from_rotm_tvec(rotm, tvec)
    xs_h = np.hstack([xs, np.ones((xs.shape[0], 1))])  # homogenize
    ys_h = (transform_expected @ xs_h.T).T
    ys = ys_h[:, :3]
    rotm, tvec = geometry.estimate_rigid_transform_rotm(xs, ys)
    transform_actual = Pose.transform_from_rotm_tvec(rotm, tvec)
    assert np.allclose(transform_actual, transform_expected)