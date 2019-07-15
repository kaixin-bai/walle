"""Tests the Virtual camera API.
"""

import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from walle.cameras.virtual import VirtualCamera


def test_virtual_iterator():
    """Loops over the iterator and displays images.
    """
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    dir_test = os.path.join(dir_name, 'virtual-cam')
    for (rgb, depth, depth_c) in VirtualCamera(dir_test):
        fig, axes = plt.subplots(1, 2)
        for ax, im in zip(axes, [rgb, depth]):
            ax.imshow(im)
            ax.axis('off')
        plt.show()
        plt.close()
