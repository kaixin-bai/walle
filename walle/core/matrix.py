"""Rotation matrices.
"""

import numpy as np


class RotationMatrix(object):
  """A convenience class for dealing with 3-D rotation matrices.

  References:
    .. [3] James Arvo, "Fast Random Rotation Matrices", The Graphics Gems Series, III.4
  """
  @classmethod
  def rotx(cls, theta):
    """Generates a `3x3` rotation matrix about the x-axis.

    Args:
      theta (:obj:`float`): The angle of rotation in radians.

    Returns:
      :obj:`ndarray`: A 3x3 rotation matrix about the x-axis.
    """
    return np.array([
      [1, 0, 0],
      [0, np.cos(theta), -np.sin(theta)],
      [0, np.sin(theta), np.cos(theta)],
    ], dtype="float64")

  @classmethod
  def roty(cls, theta):
    """Generates a `3x3` rotation matrix about the y-axis.

    Args:
      theta (:obj:`float`): The angle of rotation in radians.

    Returns:
      :obj:`ndarray`: A 3x3 rotation matrix about the y-axis.
    """
    return np.array([
      [np.cos(theta), 0, np.sin(theta)],
      [0, 1, 0],
      [-np.sin(theta), 0, np.cos(theta)],
    ], dtype="float64")

  @classmethod
  def rotz(cls, theta):
    """Generates a `3x3` rotation matrix about the z-axis.

    Args:
      theta (:obj:`float`): The angle of rotation in radians.

    Returns:
      :obj:`ndarray`: A 3x3 rotation matrix about the z-axis.
    """
    return np.array([
      [np.cos(theta), -np.sin(theta), 0],
      [np.sin(theta), np.cos(theta), 0],
      [0, 0, 1],
    ], dtype="float64")

  @classmethod
  def householder(cls, v):
    """Constructs a Householder reflection from a Householder vector.

    Args:
      v (:obj:`ndarray`): A Householder vector of length 3.

    Returns:
      :obj:`ndarray`: A 3x3 reflection matrix.
    """
    return np.eye(3) - (2 * np.dot(v, v.T))

  @classmethod
  def random(cls):
    """Fast random rotation matrix generation as described in [3]_.

    Returns:
      :obj:`ndarray`: A random 3x3 rotation matrix.
    """
    x0, x1, x2 = np.random.uniform(size=3)  # sample three uniform rvs
    theta = 2 * np.pi * x0  # pick a rotation about the pole
    phi = 2 * np.pi * x1  # pick a direction to flip the pole
    z = x2  # pick the amount of pole deflection
    # construct reflection vector
    v = np.array([
      [np.cos(phi) * np.sqrt(z)],
      [np.sin(phi) * np.sqrt(z)],
      [np.sqrt(1 - z)],
    ])
    rotm_refl = RotationMatrix.householder(v)  # construct Householder reflection
    rotm_z = RotationMatrix.rotz(theta)  # create rotation about z-axis
    rotm = -rotm_refl @ rotm_z  # randomly rotate about z, then reflect north pole
    return rotm