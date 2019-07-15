"""Camera constants.
"""

import enum


class D415(enum.Enum):
    FPS_H = 15
    FPS_L = 30
    HEIGHT_H = 720
    WIDTH_H = 1280
    HEIGHT_L = 480
    WIDTH_L = 640
    MIN_DEPTH = 0.31
    MAX_DEPTH = 10