import cv2
import matplotlib.pyplot as plt
import numpy as np

from walle.pointcloud import PointCloud


if __name__ == "__main__":
  print("Reading data...")
  cam_intr = np.loadtxt("camera_intrinsics.txt", delimiter=' ')
  cam_pose = np.loadtxt("camera_pose.txt", delimiter=' ')
  depth_im = cv2.imread("depth.png", -1).astype(float) / 10000
  color_im = cv2.cvtColor(cv2.imread("color.png"), cv2.COLOR_BGR2RGB)

  pc = PointCloud(color_im, depth_im, cam_intr)
  pc.make_pointcloud(cam_pose, depth_trunc=1.7, trim=True)
  pc.view_point_cloud()
  pc.view_imgs(figsize=None)

  # view_bounds = np.array([[0.35,0.87],[-0.1,0.22],[-0.343,0.0]])
  pc.make_heightmap(cam_pose, np.asarray([[0.15, 1.2], [-0.3, 0.4], [-5, 0]]), 0.002, -2)
  pc.view_height_map(figsize=None)

  print(pc.color_im.shape)
  print(pc.depth_im.shape)