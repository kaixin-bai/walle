# walle

walle is a general-purpose library for *robotics research* in the context of deep learning.

## Installation

You can install walle with pip:

```
pip install walle
```

## APIs

Walle features a growing set of submodules:

- `core`: a unified data structure for dealing with position and orientation in 3-D space.
- `cameras`: an extendable API for streaming data from RGB-D cameras. Currently supports the RealSense D415 camera.
- `pointcloud`: an API for transforming to and from RGB-D images, point clouds and orthographic height maps.
- `utils`: miscellaneous classes and functions useful for day to day research.

## 补充
“walle/walle/pointcloud/”中介绍了生成height map的方法
