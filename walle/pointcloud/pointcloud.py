"""Abstraction for 3-D point clouds.
"""

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


class PointCloud(object):
    """A 3-D point cloud.
    """
    def __init__(self, img, depth, intrinsics):
        """Initializes the point cloud.

        Args:
            img: (ndarray) An rgb image of shape (H, W, 3) or a
                grayscale image of shape (H, W).
            depth: (ndarray) A depth image of shape (H, W). The
                depth values are expected to be in meters.
            intrinsics: (ndarray) The camera intrinsics as a numpy
                array of shape (3, 3).
        """
        self._check_valid_img(img, depth)
        self._check_valid_intr(intrinsics)

        self._img = np.array(img, copy=True)
        self._depth = np.array(depth, copy=True)
        self._intr = intrinsics

        if img.ndim == 3:
            self._is_color = True
            self._height, self._width, self._channels = img.shape
        else:
            self._is_color = False
            self._height, self._width = img.shape

        self._preprocess_img()

        self._point_cloud = None
        self._height_map_color = None
        self._height_map_depth = None

    def _check_valid_img(self, img, depth):
        """Checks that the input data is valid.
        """
        color_cond = img.ndim == 3 and img.shape[-1] == 3
        gray_cond = img.ndim == 2
        depth_cond = img.shape[:2] == depth.shape
        assert (color_cond or gray_cond) and depth_cond

    def _check_valid_intr(self, intr):
        """Checks that the intrinsics are valid.
        """
        assert intr.shape == (3, 3)

    def _preprocess_img(self):
        """Pre-proceses the color or gray image.
        """
        if self._img.max() > 1:
            self._img = (self._img / 255.).astype("float32")

    @classmethod
    def from_point_cloud(cls, xyzrgb, intrinsics, im_shape, extrinsics=None):
        msg = "[!] Invalid image shape, product must equal point cloud length."
        assert len(xyzrgb) == np.prod(im_shape), msg
        color_im, depth_im = PointCloud.make_color_depth(xyzrgb, intrinsics, im_shape, extrinsics)
        inst = cls(color_im, depth_im, intrinsics)
        inst._point_cloud = xyzrgb
        inst._extr = extrinsics
        return inst

    def make_pointcloud(self, extrinsics=None, depth_trunc=3., trim=True):
        """Creates a 3-D point cloud.

        Args:
            extrinsics: (ndarray) The extrinsics of the camera
                of shape (4, 4).
            depth_trunc: (float) Depth values greater than this
                value are invalidated, i.e. set to NaN.
            trim: (bool) Whether to eliminate points with
                invalid depth values. This is useful for
                trimming the size of the point cloud.
        """
        extrinsics = np.eye(4) if extrinsics is None else extrinsics
        self._extr = extrinsics
        cc, rr = np.meshgrid(np.arange(self._width), np.arange(self._height), sparse=True)
        valid = (self._depth > 0) & (self._depth < depth_trunc)
        z = np.where(valid, self._depth, np.nan)
        x = np.where(valid, z * (cc - self._intr[0, 2]) / self._intr[0, 0], 0)
        y = np.where(valid, z * (rr - self._intr[1, 2]) / self._intr[1, 1], 0)
        if self._is_color:
            color = self._img.copy().transpose([2, 0, 1])
        else:
            color = [self._img.copy()]
        self._point_cloud = np.vstack([e.flatten() for e in [x, y, z, *color]]).T
        if not np.allclose(extrinsics, np.eye(4)):
            self._point_cloud = PointCloud.transform_point_cloud(self._point_cloud, extrinsics)
        if trim:
            self._point_cloud = self._point_cloud[~np.isnan(self._point_cloud[:, 2])]

    def make_heightmap(self, extrinsics, view_bounds, pixel_size, zero_level):
        """Returns a top-down orthographic heightmap image from the point cloud.

        Args:
            extrinsics: (ndarray) The extrinsics of the camera of shape (4, 4).
            view_bounds: (ndarray) The bounds of the heightmap region
                in 3D space in world coordinates of shape (3, 2).
                The rows are the [x, y, z] values and the columns
                are the respective [min, max] values.
            pixel_size: (float) A value defining the size of each pixel in meters,
                i.e. the heightmap resolution.
            zero_level: (float) A value defining the z-coordinate
                of the zero-level, i.e. the bottom of the heightmap.
        """
        if self._point_cloud is None:
            self.make_pointcloud(extrinsics)

        # compute heightmap spatial size
        heightmap_size = np.round([
            (view_bounds[1, 1] - view_bounds[1, 0]) / pixel_size,
            (view_bounds[0, 1] - view_bounds[0, 0]) / pixel_size,
        ]).astype(int)

        # figure out which indices are valid
        x_cond = np.logical_and(
            self._point_cloud[:, 0] >= view_bounds[0, 0],
            self._point_cloud[:, 0] < view_bounds[0, 1],
        )
        y_cond = np.logical_and(
            self._point_cloud[:, 1] >= view_bounds[1, 0],
            self._point_cloud[:, 1] < view_bounds[1, 1],
        )
        z_cond = np.logical_and(
            self._point_cloud[:, 2] >= view_bounds[2, 0],
            self._point_cloud[:, 2] < view_bounds[2, 1],
        )
        heightmap_valid_ind = reduce(np.logical_and, [x_cond, y_cond, z_cond])

        # remove invalid indices
        point_cloud_valid = self._point_cloud[heightmap_valid_ind]
        points = point_cloud_valid[:, :3]
        color = point_cloud_valid[:, 3:]

        # sort points by z value
        sort_z_ind = np.argsort(points[:, 2])
        points_sorted = points[sort_z_ind]
        color_sorted = color[sort_z_ind]

        # backproject pointcloud onto heightmap
        heightmap_pixel_x = np.round((points_sorted[:, 0] - view_bounds[0, 0]) / pixel_size).astype(int)
        heightmap_pixel_y = np.round((points_sorted[:, 1] - view_bounds[1, 0]) / pixel_size).astype(int)

        # clip to ensure within image bounds
        heightmap_pixel_x = np.clip(heightmap_pixel_x, 0, heightmap_size[1]-1)
        heightmap_pixel_y = np.clip(heightmap_pixel_y, 0, heightmap_size[0]-1)

        # get height values from z values minus zero level
        self._height_map_depth = np.zeros(heightmap_size)
        self._height_map_depth[heightmap_pixel_y, heightmap_pixel_x] = points_sorted[:, 2]
        self._height_map_depth = self._height_map_depth - zero_level
        self._height_map_depth[self._height_map_depth < 0] = 0
        self._height_map_depth[self._height_map_depth == -zero_level] = 0

        # finally map the colors
        num_channels = 3 if self._is_color else 1
        self._height_map_color = np.zeros((heightmap_size[0], heightmap_size[1], num_channels), dtype="uint8")
        for c in range(num_channels):
            self._height_map_color[heightmap_pixel_y, heightmap_pixel_x, c] = color_sorted[:, c] * 255
        self._height_map_color = self._height_map_color.squeeze()

    @staticmethod
    def transform_point_cloud(point_cloud, transforms):
        """Applies a rigid transform to a point cloud.

        Args:
            point_cloud: (ndarray) The point cloud of shape (N, 6) or (N, 4) or (N, 3).
            transform: (ndarray) The rigid transform of shape (4, 4). Can also
                be a list of transforms to apply sequentially.

        Returns:
            point_cloud_T: (ndarray) The transformed point cloud.
        """
        num_pts = point_cloud.shape[0]
        pts = point_cloud[:, :3]
        pts_h = np.hstack([pts, np.ones((num_pts, 1))])
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]
        for transform in transforms:
            pts_t = (transform @ pts_h.T).T
        if point_cloud.shape[1] > 3:
            clrs = point_cloud[:, 3:]
            point_cloud = np.hstack([pts_t[:, :3], clrs])
            return point_cloud
        return pts_t[:, :3]

    @staticmethod
    def points2pixels(points, intr, extr=None):
        """Projects 3-D points into 2D pixels.

        Args:
            points: (ndarray) The xyz points of shape (N, 3).
            intr: (ndarray) The camera intrinsics of shape (3, 3).
            extr: (ndarray) The camera extrinsics of shape (4, 4).

        Returns:
            uv: (ndarray) The uv pixels of shape (N, 2).
        """
        extr = np.eye(4) if extr is None else extr
        cx, cy = intr[0, 2], intr[1, 2]
        fx, fy = intr[0, 0], intr[1, 1]
        points = PointCloud.transform_point_cloud(points, np.linalg.inv(extr))
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        v = np.round((x * fx / z) + cx).astype("int")
        u = np.round((y * fy / z) + cy).astype("int")
        uv = np.vstack([u, v]).T
        return uv

    @staticmethod
    def make_color_depth(xyzrgb, intrinsics, im_shape, extrinsics):
        """Creates a color and depth image from a 3-D point cloud.
        """
        depth_im = np.zeros(im_shape)
        color_im = np.zeros((*im_shape, 3))
        pts = xyzrgb[:, :3]
        clrs = xyzrgb[:, 3:]
        uv = PointCloud.points2pixels(pts, intrinsics, extrinsics)
        valid_idx = (uv[:, 0] >= 0) & (uv[:, 0] < im_shape[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < im_shape[1])
        uv = uv[valid_idx]
        clrs = clrs[valid_idx]
        depth = pts[valid_idx, 2]
        depth_im[uv[:, 0], uv[:, 1]] = depth
        color_im[uv[:, 0], uv[:, 1]] = clrs
        return color_im, depth_im

    def view_point_cloud(self, frame=True):
        """Draws the point cloud in Open3D.

        Args:
            frame: (bool) Whether to plot the xyz coordinate frame.
        """
        # remove NaNs in case point cloud has not been trimmed
        pc = self._point_cloud[~np.isnan(self._point_cloud[:, 2])]
        pts = pc[:, :3].copy().astype(np.float64)
        if pc.shape[1] > 4:
            clrs = pc[:, 3:].copy().astype(np.float64)
        else:
            clrs = np.repeat((pc[:, 3:].copy()).astype(np.float64), 3, axis=1)
        o3d_pc = [o3d.geometry.PointCloud()]
        o3d_pc[0].points = o3d.utility.Vector3dVector(pts)
        o3d_pc[0].colors = o3d.utility.Vector3dVector(clrs)
        if frame:
            o3d_pc.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
        o3d.visualization.draw_geometries(o3d_pc)

    def view_imgs(self, figsize=(10, 10)):
        """Displays the color and depth images side by side.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(self._img, cmap=None if self._is_color else "gray")
        axes[1].imshow(self._depth)
        for ax in axes:
            ax.axis('off')
        plt.show()

    def view_height_map(self, figsize=(10, 10)):
        """Displays the color and depth height maps side by side.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(self._height_map_color, cmap=None if self._is_color else "gray")
        axes[1].imshow(self._height_map_depth)
        for ax in axes:
            ax.axis('off')
        plt.show()

    @property
    def point_cloud(self):
        if self._point_cloud is None:
            self.make_pointcloud()
        return self._point_cloud

    @property
    def height_map_color(self):
        if self._height_map_color is None:
            raise AttributeError("[!] You must first create the heightmap.")
        return self._height_map_color

    @property
    def height_map_depth(self):
        if self._height_map_depth is None:
            raise AttributeError("[!] You must first create the heightmap.")
        return self._height_map_depth

    @property
    def shape(self):
        if self._point_cloud is None:
            raise ValueError("[!] You must first create the point cloud.")
        return self._point_cloud.shape
