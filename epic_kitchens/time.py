""" Functions for converting between frames and timestamps """
import numpy as np

_MINUTES_TO_SECONDS = 60
_HOURS_TO_SECONDS = 60 * 60


def timestamp_to_seconds(timestamp: str) -> float:
    """ Convert a timestamp into total number of seconds

    Args:
        timestamp: formatted as "HH:MM:SS[.FractionalPart]"

    Returns:
        ``timestamp`` converted to seconds

    Examples:
        >>> timestamp_to_seconds("00:00:00")
        0.0
        >>> timestamp_to_seconds("00:00:05")
        5.0
        >>> timestamp_to_seconds("00:00:05.5")
        5.5
        >>> timestamp_to_seconds("00:01:05.5")
        65.5
        >>> timestamp_to_seconds("01:01:05.5")
        3665.5
    """
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * _HOURS_TO_SECONDS + minutes * _MINUTES_TO_SECONDS + seconds
    return total_seconds


def seconds_to_timestamp(total_seconds: float) -> str:
    """ Convert seconds into a timestamp

    Args:
        total_seconds: time in seconds

    Returns:
        timestamp representing ``total_seconds``

    Examples:
        >>> seconds_to_timestamp(1)
        '00:00:1.000'
        >>> seconds_to_timestamp(1.1)
        '00:00:1.100'
        >>> seconds_to_timestamp(60)
        '00:01:0.000'
        >>> seconds_to_timestamp(61)
        '00:01:1.000'
        >>> seconds_to_timestamp(60 * 60 + 1)
        '01:00:1.000'
        >>> seconds_to_timestamp(60 * 60  + 60 + 1)
        '01:01:1.000'
        >>> seconds_to_timestamp(1225.78500002)
        '00:20:25.785'
    """
    ss = total_seconds % 60
    mm = np.floor((total_seconds / 60) % 60)
    hh = np.floor((total_seconds / (60 * 60)))
    return "{:02.0f}:{:02.0f}:{:0.3f}".format(hh, mm, ss)


def timestamp_to_frame(timestamp: str, fps: float) -> int:
    """ Convert timestamp to frame number given the FPS of the extracted frames
    Args:
        timestamp: formatted as "HH:MM:SS[.FractionalPart]"
        fps: frames per second

    Returns:
        frame corresponding to a specific

    Examples:
        >>> timestamp_to_frame("00:00:00", 29.97)
        1
        >>> timestamp_to_frame("00:00:01", 29.97)
        29
        >>> timestamp_to_frame("00:00:01", 59.94)
        59
        >>> timestamp_to_frame("00:01:00", 60)
        3600
        >>> timestamp_to_frame("01:00:00", 60)
        216000
    """

    total_seconds = timestamp_to_seconds(timestamp)
    if total_seconds == 0:
        return 1
    else:
        return int(np.floor(total_seconds * fps))


def flow_frame_count(rgb_frame: int, stride: int, dilation: int) -> int:
    """ Get the number of frames in a optical flow segment given the number of frames in the
    corresponding rgb segment from which the flow was extracted with parameters
    ``(stride, dilation)``

    Args:
        rgb_frame: RGB Frame number
        stride: Stride used in extracting optical flow
        dilation: Dilation used in extracting optical flow

    Returns:
       The number of optical flow frames

    Examples:
        >>> flow_frame_count(6, 1, 1)
        5
        >>> flow_frame_count(6, 2, 1)
        3
        >>> flow_frame_count(6, 1, 2)
        4
        >>> flow_frame_count(6, 2, 2)
        2
        >>> flow_frame_count(6, 3, 1)
        2
        >>> flow_frame_count(6, 1, 3)
        3

        >>> flow_frame_count(7, 1, 1)
        6
        >>> flow_frame_count(7, 2, 1)
        3
        >>> flow_frame_count(7, 1, 2)
        5
        >>> flow_frame_count(7, 2, 2)
        3
        >>> flow_frame_count(7, 3, 1)
        2
        >>> flow_frame_count(7, 1, 3)
        4
    """
    return int(np.ceil((float(rgb_frame) - dilation) / stride))
