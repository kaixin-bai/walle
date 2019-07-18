import logging
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from walle.cameras import constants, utils
from walle.cameras.base import Camera


class RealSenseD415(Camera):
    """Interface for the Intel RealSense D415 depth camera.

    Attributes:
        serial (str): The serial number of the sensor. This is so we can
            differentiate between multiple connected devices. If no serial
            number is provided, we try to connect to a random one.
        resolution (str): Whether to use a `high` resolution of (720, 1280)
            or `low` resolution of (480, 640).
        calib_file (str): A json file containing values for calibrated intrinsics,
            extrinsics and depth scale factor.
    """
    def __init__(self, serial=None, resolution="high", calib_file=None):
        self._is_start = False
        self._frame_id = 0
        self._auto_exp_skip = 30
        self.calib_file = calib_file

        msg = "`resolution` can only be [`low` or `high`]"
        assert resolution in ['low', 'high'], msg

        # query any connected realsense
        if serial is None:
            serials = utils.rs_discover_cams()
            if serials:
                self._serial = serials[0]
            else:
                raise ValueError("Could not find connected camera. Ensure USB 3.0 is used.")
        else:
            self._serial = serial

        if resolution == "low":
            self._height = constants.D415.HEIGHT_L.value
            self._width = constants.D415.WIDTH_L.value
            self._fps = constants.D415.FPS_L.value
        else:
            self._height = constants.D415.HEIGHT_H.value
            self._width = constants.D415.WIDTH_H.value
            self._fps = constants.D415.FPS_H.value
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

            # initialize camera parameters
            self._init_params()

            # skip few frames to give auto-exposure a chance to settle
            for _ in range(self._auto_exp_skip):
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
        self._cfg.enable_device(self._serial)

        # configure color stream
        self._cfg.enable_stream(
            rs.stream.color,
            self._width,
            self._height,
            rs.format.rgb8,
            self._fps,
        )

        # configure depth stream
        self._cfg.enable_stream(
            rs.stream.depth,
            self._width,
            self._height,
            rs.format.z16,
            self._fps,
        )

    def set_auto_exposure(self, val):
        """Manually set auto-exposure value.
        """
        depth_sensor = self._profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.enable_auto_exposure, np.clip(val, 0, 1))

    def _get_factory_intrinsics(self):
        """Queries the factory intrinsics from the device.
        """
        strm = self._profile.get_stream(rs.stream.color)
        vals = strm.as_video_stream_profile().get_intrinsics()
        intr = utils.intrinsics_from_vals(
            vals.fx,
            vals.fy,
            vals.ppx,
            vals.ppy,
        )
        return intr

    def _set_default_params(self):
        """Sets default camera parameters.

        This is called if the calibration file cannot be read
        or isn't provided.
        """
        self._intr = self._get_factory_intrinsics()
        self._extr = np.zeros((1, 6))
        self._dist = np.zeros((1, 5))
        self._depth_scale = 1

    def _init_params(self):
        """Initializes camera parameters.
        """
        if self.calib_file is not None:
            try:
                with open(self.calib_file, 'r') as fp:
                    params = json.load(fp)
                self._intr = utils.intrinsics_from_vals(
                   params['intrinsics'][0],
                   params['intrinsics'][1],
                   params['intrinsics'][2],
                   params['intrinsics'][3],
                )
                self._dist = np.array([params['distortion']])
                self._extr = np.array([params['extrinsics']])
                self._depth_scale = params['depth_scale']
            except:
                self._set_default_params()
        else:
            self._set_default_params()

    def _post_process(self, color_frame, depth_frame, filter_depth):
        if filter_depth:
            depth_frame = self._filter_depth(depth_frame)

        # colorize depth frame for visualization
        depth_c_frame = self._colorizer.colorize(depth_frame)

        color = self._to_numpy(color_frame, np.uint8)
        depth = self._to_numpy(depth_frame, np.float32)
        depth_c = self._to_numpy(depth_c_frame, np.uint8)

        # convert depth to meters
        depth *= (1e-3 * self._depth_scale)

        return color, depth, depth_c

    def _to_numpy(self, frame, dtype):
        arr = np.asanyarray(frame.get_data(), dtype=dtype)
        return arr

    def _filter_depth(self, depth, temporal=False):
        out = self._spatial_filter.process(depth)
        if temporal:
            out = self._temporal_filter.process(out)
        return self._hole_filling.process(out)

    def get_frame(self, aligned=True, undistort=False, filter_depth=True, colorized=False):
        frames = self._pipe.wait_for_frames()
        if aligned:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            logging.warning("[!] Could not retrieve depth or color frame.")
            return None

        color, depth, depth_c = self._post_process(color_frame, depth_frame, filter_depth)

        if undistort:
            H, W = color.shape
            new_intr, _ = cv2.getOptimalNewCameraMatrix(self._intr, self._dist, (W, H), 1)
            map_x, map_y = cv2.initUndistortRectifyMap(self._intr, self._dist, None, new_intr, (W, H), cv2.CV_32FC1)
            color = cv2.remap(color, map_x, map_y, cv2.INTER_LINEAR)

        ret = {
            "frame_id": self._frame_id,
            "color": color,
            "depth": depth,
        }
        if colorized:
            ret["depth_c"] = depth_c

        self._frame_id += 1
        return ret

    def view_feed(self):
        """Shows live video feeds using cv2 pane.
        """
        current_milli_time = lambda: int(round(time.time() * 1000))
        cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
        frame_id_prev = self._frame_id
        timestamp_prev = current_milli_time()
        fps = self._fps
        while (True):
            frames = self._pipe.wait_for_frames()
            frames = self._align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            timestamp_curr = current_milli_time()
            if timestamp_curr - timestamp_prev > 1000:
                fps = 1000 * (self._frame_id - frame_id_prev) / (timestamp_curr - timestamp_prev)
                timestamp_prev = timestamp_curr
                frame_id_prev = self._frame_id
            depth = np.asanyarray(depth_frame.get_data())
            color_rgb = np.asanyarray(color_frame.get_data())
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            depth_c = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_bgr, depth_c))
            cv2.putText(
                images,
                "{}".format(int(fps)),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )
            cv2.imshow('video', images)
            self._frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)

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
        color_frame = frames.get_color_frame()

        # filter the depth frames
        for depth_frame in depth_burst:
            depth_frame = self._filter_depth(depth_frame, temporal=True)

        depth_c_frame = self._colorizer.colorize(depth_frame)

        color = self._to_numpy(color_frame, np.uint8)
        depth = self._to_numpy(depth_frame, np.float32)
        depth_c = self._to_numpy(depth_c_frame, np.uint8)

        # convert depth to meters
        depth *= (1e-3 * self._depth_scale)

        ret = {
            "frame_id": [self._frame_id + i for i in range(num_frames)],
            "color": color,
            "depth": depth,
        }
        if colorized:
            ret["depth_c"] = depth_c

        self._frame_id += num_frames
        return ret

    def get_median(self, num_frames, aligned=True, colorized=False):
        depth_burst = []
        while (len(depth_burst) < num_frames):
            frames = self._pipe.wait_for_frames()
            if aligned:
                frames = self._align.process(frames)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_burst.append(depth_frame)
        color_frame = frames.get_color_frame()
        depth_c_frame = self._colorizer.colorize(depth_burst[-1])

        depths = []
        for depth_frame in depth_burst:
            d_arr = self._to_numpy(depth_frame, np.float32)
            d_arr *= (1e-3 * self._depth_scale)
            depths.append(d_arr)
        depth = np.median(np.stack(depths), axis=0)
        depth_c = self._to_numpy(depth_c_frame, np.uint8)
        color = self._to_numpy(color_frame, np.uint8)

        ret = {
            "frame_id": [self._frame_id + i for i in range(num_frames)],
            "color": color,
            "depth": depth,
        }
        if colorized:
            ret["depth_c"] = depth_c

        self._frame_id += num_frames
        return ret

    @property
    def intrinsics(self):
        return self._intr

    @property
    def extrinsics(self):
        return self._extr

    @property
    def depth_scale(self):
        return self._depth_scale

    @property
    def distortion(self):
        return self._dist

    @property
    def resolution(self):
        return self._resolution

    @property
    def serial(self):
        return self._serial
