"""Tests the RealSense camera API.
"""

import pytest

from walle.cameras.realsense import RealSenseD400


def test_video_feed():
    """Tests that the video feed starts and stops cleanly."""
    ids = RealSenseD400.discover_cams()
    with RealSenseD400(id=ids[0]) as cam:
        cam.view_feed()


def test_intrinsics():
    """Tests that the video feed starts and stops cleanly."""
    ids = RealSenseD400.discover_cams()
    with RealSenseD400(id=ids[0]) as cam:
        print(cam.intrinsics)
