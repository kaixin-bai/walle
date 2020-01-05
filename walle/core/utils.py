import collections

import numpy as np


def is_iterable(x):
  """Tests that `x` is an iterable.

  In particular, this method is useful for
  checking that an input is either a list
  or a numpy ndarray.
  """
  is_string = isinstance(x, str)
  is_iter = isinstance(x, collections.abc.Iterable)
  return is_iter and not is_string


def is_1d_iterable(x):
  """Tests that `x` is a 1-D list of reals.
  """
  # transform into a numpy array
  try:
    x = np.asarray(x, dtype="float64")
  except BaseException:
    return False
  # squeeze any extraneous dimensions
  x = x.squeeze()
  return x.ndim == 1


def is_single_vector(x):
  """Tests that `x` is a single 3-D vector.
  """
  try:
    x = np.asarray(x).squeeze()
  except BaseException:
    return False
  is_len_3 = len(x) == 3
  return is_1d_iterable(x) and is_len_3


def is_batch_vectors(x):
  """Tests that `x` is a batch of 3-D vectors.
  """
  x = np.asarray(x).squeeze()
  return x.ndim == 2 and x.shape[1] == 3