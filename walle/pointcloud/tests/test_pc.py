import cv2
import matplotlib.pyplot as plt
import numpy as np

from walle.pointcloud import PointCloud


if __name__ == "__main__":
    print("Reading data...")
    cam_intr = np.loadtxt("camera_intrinsics.txt", delimiter=' ')
    cam_pose = np.loadtxt("camera_pose.txt", delimiter=' ')
    depth_im = cv2.imread("depth.png", -1).astype(float)
    depth_im /= 10000
    color_im = cv2.cvtColor(cv2.imread("color.png"), cv2.COLOR_BGR2RGB)

    pc = PointCloud(color_im, depth_im, cam_intr)
    pc.make_pointcloud(cam_pose, depth_trunc=10, trim=True)
    # pc.view_point_cloud()
    # pc.view_imgs()

    # view_bounds = np.array([[0.35,0.87],[-0.1,0.22],[-0.343,0.0]])
    # pc.make_heightmap(cam_pose, np.asarray([[0.15, 1.2], [-0.3, 0.4], [-5, 0]]), 0.002, -2)
    # pc.view_height_map()

    # point_cloud = pc.point_cloud
    # assert point_cloud.shape[0] == np.prod(depth_im.shape)
    # pc = PointCloud.from_point_cloud(point_cloud, cam_intr, depth_im.shape, cam_pose)