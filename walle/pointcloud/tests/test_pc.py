import cv2
import matplotlib.pyplot as plt
import numpy as np

from walle.pointcloud import PointCloud


if __name__ == "__main__":
    print("Reading data...")
    cam_intr = np.loadtxt("camera-intrinsics.txt", delimiter=' ')
    cam_pose = np.loadtxt("camera-pose.txt", delimiter=' ')
    depth_im = cv2.imread("depth.png", -1).astype(float)
    depth_im /= 1000
    depth_im[depth_im == 65.535] = 0
    color_im = cv2.cvtColor(cv2.imread("color.jpg"), cv2.COLOR_BGR2RGB)

    pc = PointCloud(color_im, depth_im, cam_intr)
    pc.make_pointcloud(extrinsics=cam_pose, depth_trunc=500., trim=False)
    pc.view_point_cloud()

    point_cloud = pc.point_cloud
    assert point_cloud.shape[0] == np.prod(depth_im.shape)

    pc = PointCloud.from_point_cloud(point_cloud, cam_intr, depth_im.shape, cam_pose)
    pc.view_point_cloud()
    pc.view_imgs()