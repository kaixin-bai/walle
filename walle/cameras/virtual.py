import glob
import os

import numpy as np

from PIL import Image

from walle.cameras import constants


class VirtualCamera(object):
  """A virtual RGB-D camera.

  This is a dummy camera class that just reads and returns
  pre-captured images stored in a folder. The returned images
  are strictly read-only, i.e. it is assumed that the stored
  images have already been pre-processed.

  This class is inspired by [1].

  Attributes:
    dir_path: (str) the path to the folder. It should
      contain two mandatory subfolders `color`, `depth`
      and an optional `depth_c` subfolder. It should also
      contain a json file with the camera intrinsics.

  References:
    .. [1]  BerkeleyAutomation, `perception` module,
      https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_sensor.py
  """
  def __init__(self, dir_path):
    self._is_start = False
    self.dir_path = dir_path

    self._set_intrinsics()

    self._color_dir = os.path.join(dir_path, 'color', '')
    self._depth_dir = os.path.join(dir_path, 'depth', '')
    self._depth_c_dir = os.path.join(dir_path, 'depth_c', '')

    self._color_filenames = sorted(glob.glob(
      self._color_dir + '*.{}'.format(constants.COLOR_EXT)
    ))
    self._depth_filenames = sorted(glob.glob(
      self._depth_dir + '*.{}'.format(constants.DEPTH_EXT)
    ))
    if os.path.exists(self._depth_c_dir):
      self._depth_c_filenames = sorted(glob.glob(
        self._depth_c_dir + '*.{}'.format(constants.DEPTH_C_EXT)
      ))
    else:
      self._depth_c_filenames = None

    self._counter = 0
    self._num_frames = len(self._color_filenames)

  def _set_intrinsics(self):
    """Reads and stores the intrinsics matrix.
    """
    intr_file = os.path.join(self.dir_path, 'virtual.{}'.format(constants.INTRINSICS_EXT))
    self._intrinsics = np.loadtxt(intr_file)

  def __iter__(self):
    return self

  def __next__(self):
    if self._counter == self._num_frames:
      raise StopIteration
    else:
      color = self._get_color()
      depth = self._get_depth()

      if self._depth_c_filenames:
        depth_c = self._get_depth_c()
        self._counter += 1
        return (color, depth, depth_c)
      else:
        self._counter += 1
        return (color, depth)

  def _get_color(self):
    img = Image.open(self._color_filenames[self._counter])
    return np.array(img)

  def _get_depth(self):
    return np.load(self._depth_filenames[self._counter])

  def _get_depth_c(self):
    img = Image.open(self._depth_c_filenames[self._counter])
    return np.array(img)

  @property
  def intrinsics(self):
    return self._intrinsics

  @property
  def num_frames(self):
    return self._num_frames
