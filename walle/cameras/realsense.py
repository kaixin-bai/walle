import logging
import os

import cv2
import numpy as np

from walle.cameras import constants
from walle.cameras.base import Camera

try:
    import pyrealsense2 as rs
except ImportError:
    logging.error("[!] Unable to import pyrealsense2.")


class RealSenseD400(Camera):
    """Camera class for the Intel RealSense D400-Series depth cameras.

    Attributes:
        id: (str) The serial number of the sensor. This is so we can
            differentiate between multiple connected devices.
        resolution: (str) Whether to use a `high` resolution of (720, 1280)
            or `low` resolution of (480, 640).
        calib_dir: (str) The directory containing the calibration output
            files (i.e. intrinsics, extrinsics and depth scale).
    """

    FPS = 30
    HEIGHT_H = 720
    WIDTH_H = 1280
    HEIGHT_L = 480
    WIDTH_L = 640
    MIN_DEPTH = 0.31  # page 66 of datasheet
    MAX_DEPTH = 10
    AUTO_EXP_SKIP = 30

    def __init__(self, id, resolution="high", calib_dir=None): 
        self._is_start = False

        msg = "`resolution` can only be [`low` or `high`]"
        assert resolution in ['low', 'high'], msg

        if calib_dir is not None:
            self._depth_scale = np.loadtxt(os.path.join(calib_dir, "depth_scale.txt"))
            self._extrinsics = np.loadtxt(os.path.join(calib_dir, "extrinsics.txt"))
        else:
            self._depth_scale = constants.MM_2_METERS
            self._extrinsics = np.eye(4)

        self._id = id
        self._intrinsics = None

        if resolution == "low":
            self._height = RealSenseD400.HEIGHT_L
            self._width = RealSenseD400.WIDTH_L
        else:
            self._height = RealSenseD400.HEIGHT_H
            self._width = RealSenseD400.WIDTH_H
        self._resolution = (self._width, self._height)

        # pipeline and config
        self._pipe = rs.pipeline()
        self._cfg = rs.config()

        # post-processing
        self._spatial_filter = rs.spatial_filter()
        self._hole_filling = rs.hole_filling_filter()
        self._temporal_filter = rs.temporal_filter()

        # misc
        self._colorizer = rs.colorizer()
        self._align = rs.align(rs.stream.color)

    def start(self):
        try:
            self._config_pipe()
            self._profile = self._pipe.start(self._cfg)

            # store intrinsics
            self._set_intrinsics()

            # skip few frames to give auto-exposure a chance to settle
            for _ in range(RealSenseD400.AUTO_EXP_SKIP):
                self._pipe.wait_for_frames()

            self._is_start = True
        except RuntimeError as e:
            print(e)

    def stop(self):
        if not self._is_start:
            logging.warning("[!] Sensor is not on.")
            return False

        self._pipe.stop()
        self._is_start = False
        return True

    def _config_pipe(self):
        """Configures pipeline to stream color and depth.
        """
        self._cfg.enable_device(self._id)

        # configure color stream
        self._cfg.enable_stream(
            rs.stream.color,
            self._width,
            self._height,
            rs.format.rgb8,
            RealSenseD400.FPS,
        )

        # configure depth stream
        self._cfg.enable_stream(
            rs.stream.depth,
            self._width,
            self._height,
            rs.format.z16,
            RealSenseD400.FPS,
        )

    def _set_intrinsics(self):
        """Creates and stores the intrinsics matrix.
        """
        strm = self._profile.get_stream(rs.stream.color)
        intr = strm.as_video_stream_profile().get_intrinsics()
        self._intrinsics = np.eye(3)
        self._intrinsics[0, 0] = intr.fx
        self._intrinsics[1, 1] = intr.fy
        self._intrinsics[0, 2] = intr.ppx
        self._intrinsics[1, 2] = intr.ppy

    def _post_process(self, rgb_frame, depth_frame, filter_depth):
        if filter_depth:
            depth_frame = self._filter_depth(depth_frame)

        # colorize depth frame for visualization
        depth_c_frame = self._colorizer.colorize(depth_frame)

        rgb = self._to_numpy(rgb_frame, np.uint8)
        depth = self._to_numpy(depth_frame, np.float32)
        depth_c = self._to_numpy(depth_c_frame, np.uint8)

        # convert depth to meters
        depth *= self._depth_scale

        return (rgb, depth, depth_c)

    def _to_numpy(self, frame, dtype):
        arr = np.asanyarray(frame.get_data(), dtype=dtype)
        return arr

    def _filter_depth(self, depth, temporal=False):
        out = self._spatial_filter.process(depth)
        if temporal:
            out = self._temporal_filter.process(out)
        return self._hole_filling.process(out)

    def get_frame(self, aligned=True, filter_depth=True, colorized=False):
        frames = self._pipe.wait_for_frames()
        if aligned:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        rgb_frame = frames.get_color_frame()

        if not depth_frame or not rgb_frame:
            logging.warning("[!] Could not retrieve depth or color frame.")
            return (None,) * 3

        rgb, depth, depth_c = self._post_process(
            rgb_frame, depth_frame, filter_depth,
        )

        if colorized:
            return (rgb, depth, depth_c)
        else:
            return (rgb, depth)

    def view_feed(self):
        """Shows live video feeds using cv2 pane.
        """
        self.start()
        cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
        while (True):
            frames = self._pipe.wait_for_frames()
            frames = self._align.process(frames)
            depth_frame = frames.get_depth_frame()
            rgb_frame = frames.get_color_frame()
            if not depth_frame or not rgb_frame:
                continue
            depth = np.asanyarray(depth_frame.get_data())
            rgb = np.asanyarray(rgb_frame.get_data())
            depth_c = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((rgb, depth_c))
            cv2.imshow('video', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)
        self.stop()

    def get_burst(self, num_frames, aligned=True, colorized=False):
        # get the required number of frames
        depth_burst = []
        while (len(depth_burst) < num_frames):
            frames = self._pipe.wait_for_frames()
            if aligned:
                frames = self._align.process(frames)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_burst.append(depth_frame)
        rgb_frame = frames.get_color_frame()

        # filter the depth frames
        for depth_frame in depth_burst:
            depth_frame = self._filter_depth(depth_frame, temporal=True)

        depth_c_frame = self._colorizer.colorize(depth_frame)

        rgb = self._to_numpy(rgb_frame, np.uint8)
        depth = self._to_numpy(depth_frame, np.float32)
        depth_c = self._to_numpy(depth_c_frame, np.uint8)

        # convert depth to meters
        depth *= self._depth_scale

        if colorized:
            return (rgb, depth, depth_c)
        else:
            return (rgb, depth)

    def get_median(self, num_frames, aligned=True):
        depth_burst = []
        while (len(depth_burst) < num_frames):
            frames = self._pipe.wait_for_frames()
            if aligned:
                frames = self._align.process(frames)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_burst.append(depth_frame)
        rgb_frame = frames.get_color_frame()
        depths = []
        for depth_frame in depth_burst:
            depths.append(self._to_numpy(depth_frame, np.float32))
        depth = np.median(np.stack(depths), axis=0)
        rgb = self._to_numpy(rgb_frame, np.uint8)
        depth *= self._depth_scale
        return rgb, depth

    @property
    def depth_scale(self):
        return self._depth_scale

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def extrinsics(self):
        return self._extrinsics

    @property
    def resolution(self):
        return self._resolution

    @property
    def id(self):
        return self._id

    @staticmethod
    def discover_cams():
        """Returns a list of the ids of all cameras connected via USB.
        """
        ids = []
        for dev in rs.context().query_devices():
            ids.append(dev.get_info(rs.camera_info.serial_number))
        return ids
