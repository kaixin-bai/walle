"""Tests geometry methods.
"""

import pytest

import numpy as np

from walle.utils import geometry
from walle.core import RotationMatrix, Pose


def test_estimate_rotation_no_ransac():
  xs = np.random.randn(5, 3)
  rotm_expected = RotationMatrix.rotz(np.radians(np.random.uniform(0, 360)))
  ys = (rotm_expected @ xs.T).T
  transform_actual, _ = geometry.estimate_rigid_transform(xs, ys, use_ransac=False)
  rotm_actual = transform_actual[:3, :3]
  assert np.allclose(rotm_actual, rotm_expected)


def test_estimate_rotation_with_ransac():
  xs = np.random.randn(5, 3)
  rotm_expected = RotationMatrix.rotz(np.radians(np.random.uniform(0, 360)))
  ys = (rotm_expected @ xs.T).T
  transform_actual, _ = geometry.estimate_rigid_transform(xs, ys, use_ransac=True)
  rotm_actual = transform_actual[:3, :3]
  assert np.allclose(rotm_actual, rotm_expected)


def test_estimate_rigid_transform_identity_no_ransac():
  xs = np.random.randn(5, 3)
  tr_expected = np.eye(4)
  ys = xs.copy()
  tr_actual, _ = geometry.estimate_rigid_transform(xs, ys, use_ransac=False)
  assert np.allclose(tr_actual, tr_expected)


def test_estimate_rigid_transform_identity_with_ransac():
  xs = np.random.randn(5, 3)
  tr_expected = np.eye(4)
  ys = xs.copy()
  tr_actual, _ = geometry.estimate_rigid_transform(xs, ys, use_ransac=True)
  assert np.allclose(tr_actual, tr_expected)


def test_estimate_rigid_transform_rotation_and_translation_no_ransac():
  xs = np.random.randn(5, 3)
  rotm = RotationMatrix.rotz(np.radians(np.random.uniform(0, 360)))
  tvec = 5 * np.random.randn(1, 3)
  transform_expected = Pose.transform_from_rotm_tvec(rotm, tvec)
  xs_h = np.hstack([xs, np.ones((xs.shape[0], 1))])  # homogenize
  ys_h = (transform_expected @ xs_h.T).T
  ys = ys_h[:, :3]
  transform_actual, _ = geometry.estimate_rigid_transform(xs, ys, use_ransac=False)
  assert np.allclose(transform_actual, transform_expected)


def test_estimate_rigid_transform_rotation_and_translation_with_ransac():
  xs = np.random.randn(5, 3)
  rotm = RotationMatrix.rotz(np.radians(np.random.uniform(0, 360)))
  tvec = 5 * np.random.randn(1, 3)
  transform_expected = Pose.transform_from_rotm_tvec(rotm, tvec)
  xs_h = np.hstack([xs, np.ones((xs.shape[0], 1))])  # homogenize
  ys_h = (transform_expected @ xs_h.T).T
  ys = ys_h[:, :3]
  transform_actual, _ = geometry.estimate_rigid_transform(xs, ys, use_ransac=True)
  assert np.allclose(transform_actual, transform_expected)


# TODO: add test with outliers