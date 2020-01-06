import numpy as np
import skimage.io as io

from PIL import Image


def colorsave(filename, x):
  """Saves an rgb image as a 24 bit PNG.
  """
  io.imsave(filename, x, check_contrast=False)


def depthsave(filename, x):
  """Saves a depth image as a 16 bit PNG.
  """
  io.imsave(filename, (x * 1000).astype("uint16"), check_contrast=False)


def colorload(filename):
  """Loads an rgb image as a numpy array.
  """
  return np.asarray(Image.open(filename))


def depthload(filename):
  """Loads a depth image as a numpy array.
  """
  if filename.split(".")[-1] == "txt":
    x = np.loadtxt(filename)
  else:
    x = np.asarray(Image.open(filename))
  x = (x * 1e-3).astype("float32")
  return x