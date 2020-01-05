"""Useful camera functions.
"""

import numpy as np
import pyrealsense2 as rs


def rs_discover_cams():
  """Returns a list of ids of all realsense cameras connected via USB.
  """
  ids = []
  for dev in rs.context().query_devices():
    ids.append(dev.get_info(rs.camera_info.serial_number))
  return ids


def intrinsics_from_vals(fx, fy, cx, cy):
  """Creates a 3x3 intrinsics matrix from pinhole params.
  """
  intr = np.eye(3)
  intr[0, 0] = fx
  intr[1, 1] = fy
  intr[0, 2] = cx
  intr[1, 2] = cy
  return intr
