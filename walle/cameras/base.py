"""Abstraction for RGB-D sensors.
"""

from abc import ABC, abstractmethod, abstractproperty


class Camera(ABC):
    """An abstract class for interfacing with RGB-D sensors.
    """
    @abstractmethod
    def __init__(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    @abstractmethod
    def start(self):
        """Starts the sensor steams.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stops the sensor streams.
        """
        pass

    def reset(self):
        """Restarts the sensor streams.
        """
        self.stop()
        self.start()

    @abstractmethod
    def get_frame(self):
        """Grabs a frame from every enabled stream.
        """
        pass

    @abstractmethod
    def get_burst(self, num_frames):
        """Takes a series of consecutive frames.

        This is useful for applying temporal filters
        on sequences of depth images. It returns the
        last captured rgb frame and a temporally
        filtered depth frame.
        """
        pass

    @abstractmethod
    def view_feed(self):
        """Displays a live video feed.
        """
        pass

    @abstractproperty
    def depth_scale(self):
        """Returns the depth scale.
        """
        pass

    @abstractproperty
    def intrinsics(self):
        """Returns the intrinsics matrix.
        """
        pass

    @abstractproperty
    def extrinsics(self):
        """Returns the extrinsics matrix.
        """
        pass
