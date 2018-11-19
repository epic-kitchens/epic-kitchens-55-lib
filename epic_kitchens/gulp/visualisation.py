import PIL.Image
import numpy as np
from abc import ABC
from moviepy.editor import ImageSequenceClip
from typing import List, Union, Tuple

from epic_kitchens.dataset.epic_dataset import EpicVideoDataset


def _grey_to_rgb(frames: np.ndarray) -> np.ndarray:
    """
    Convert frame(s) from gray (2D array) to RGB (3D array)

    Args:
        frames: A single frame or set of frames
    """
    if not (2 <= frames.ndim <= 3):
        raise ValueError(
            "Expected frame(s) to be a 2D array (single frame) "
            "or 3D array, but was {}D".format(frames.ndim)
        )
    n_channels = 3
    return np.repeat(np.array(frames)[..., np.newaxis], n_channels, axis=-1)


def clipify_rgb(
    frames: List[PIL.Image.Image], *, fps: float = 60.
) -> ImageSequenceClip:
    """

    Args:
        frames: A list of frames
        fps: FPS of clip

    Returns
    -------
    moviepy.editor.ImageSequenceClip
        Frames concatenated into clip



    """
    if frames == 0:
        raise ValueError("Expected at least one frame, but received none")
    frames = [np.array(frame) for frame in frames]
    segment_clip = ImageSequenceClip(frames, fps=fps)
    return segment_clip


def clipify_flow(
    frames: List[PIL.Image.Image], *, fps: float = 30.
) -> ImageSequenceClip:
    """
    Destack flow frames, join them side by side and then create a clip
    for display

    Args:
        frames:
            A list of alternating :math:`u`, :math:`v` flow frames to join into a video. Even indices should
            be :math:`u` flow frames, and odd indices, :math:`v` flow frames.

        fps: float, optional
            FPS of generated :py:class:`moviepy.editor.ImageSequenceClip`
    """
    if frames == 0:
        raise ValueError("Expected at least one frame, but received none")
    frames = _grey_to_rgb(combine_flow_uv_frames(frames))
    return ImageSequenceClip(list(frames), fps=fps)


def combine_flow_uv_frames(
    uv_frames: Union[List[PIL.Image.Image], np.ndarray],
    *,
    method="hstack",
    width_axis=2
) -> np.ndarray:
    """Destack (u, v) frames and concatenate them side by side for display purposes"""
    if isinstance(uv_frames[0], PIL.Image.Image):
        uv_frames = list(map(np.array, uv_frames))
    u_frames = np.array(uv_frames[::2])
    v_frames = np.array(uv_frames[1::2])

    return hstack_frames(u_frames, v_frames, width_axis)


def hstack_frames(*frame_sequences: np.ndarray, width_axis: int = 2) -> np.ndarray:
    return np.concatenate(frame_sequences, axis=width_axis)


class Visualiser(ABC):
    def __init__(self, dataset: EpicVideoDataset) -> None:
        self.dataset = dataset

    def show(self, uid: Union[int, str], **kwargs) -> ImageSequenceClip:
        """
        Show the given video corresponding to ``uid`` in a HTML5 video element.

        Args:
            uid: UID of video segment
            fps (float, optional): FPS of video sequence
        """
        frames = self.dataset.load_frames(self.dataset[uid])
        clip = self._clipify_frames(frames, **kwargs)
        clip.ipython_display()
        return clip

    def _clipify_frames(self, frames: List[PIL.Image.Image], fps: float = 60.):
        raise NotImplementedError()


class RgbVisualiser(Visualiser):
    """Visualiser for video dataset containing RGB frames"""

    def _clipify_frames(self, frames: List[PIL.Image.Image], fps: float = 60.):
        return clipify_rgb(frames, fps=fps)


class FlowVisualiser(Visualiser):
    """Visualiser for video dataset containing optical flow :math:`(u, v)` frames"""

    def _clipify_frames(self, frames: List[PIL.Image.Image], fps: float = 30):
        return clipify_flow(frames, fps=fps)
