# walle

walle is a general-purpose library for *robotics research* in the context of deep learning.

## Installation

You can install walle with pip:

```
pip install walle
```

## APIs

Walle features a growing set of submodules:

- `core`: A unified data structure for dealing with position and orientation in 3-D space.
- `cameras`: An extendable API for streaming data from RGB-D cameras. Currently supports the RealSense D415 camera.
- `pointcloud`: An API for transforming to and from RGB-D images and point cloud
- `utils`: Miscellaneous classes and functions useful for day to day research.