import PIL.Image
import numpy as np
from abc import ABC
from moviepy.editor import ImageSequenceClip
from typing import List, Union, Tuple

from epic_kitchens.dataset.epic_dataset import EpicVideoDataset


def _grey_to_rgb(frames: np.array) -> np.array:
    """
    Convert frame(s) from gray (2D array) to RGB (3D array)
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
    if frames == 0:
        raise ValueError("Expected at least one frame, but received none")
    frames = [np.array(frame) for frame in frames]
    segment_clip = ImageSequenceClip(frames, fps=fps)
    return segment_clip


def clipify_flow(
    frames: List[PIL.Image.Image], *, fps: float = 30.
) -> ImageSequenceClip:
    """
    Destack flow frames, join them side by side and then create an ImageSequenceClip
    for display
    """
    if frames == 0:
        raise ValueError("Expected at least one frame, but received none")
    frames = _grey_to_rgb(combine_flow_uv_frames(frames))
    return ImageSequenceClip(list(frames), fps=fps)


def combine_flow_uv_frames(
    uv_frames: Union[List[PIL.Image.Image], np.array], *, method="hstack", width_axis=2
) -> np.array:
    """
    Destack (u, v) frames and concatenate them side by side for display purposes
    """
    if isinstance(uv_frames[0], PIL.Image.Image):
        uv_frames = list(map(np.array, uv_frames))
    u_frames = np.array(uv_frames[::2])
    v_frames = np.array(uv_frames[1::2])

    return hstack_frames(u_frames, v_frames, width_axis)


def hstack_frames(*frame_sequences: np.array, width_axis: int = 2) -> np.array:
    return np.concatenate(frame_sequences, axis=width_axis)


class Visualiser(ABC):
    def __init__(self, dataset: EpicVideoDataset) -> None:
        self.dataset = dataset

    def show(self, uid: Union[int, str], **kwargs) -> ImageSequenceClip:
        """
        Show the given video corresponding to ``uid`` in a HTML5 video element.

        Parameters
        ----------
        uid: UID of video segment
        fps (float): optional,

        Returns
        -------
        ImageSequenceClip of the sequence
        """
        frames = self.dataset.load_frames(self.dataset[uid])
        clip = self._clipify_frames(frames, **kwargs)
        clip.ipython_display()
        return clip

    def _clipify_frames(self, frames: List[PIL.Image.Image], fps: float = 60.):
        raise NotImplementedError()


class RgbVisualiser(Visualiser):
    def _clipify_frames(self, frames: List[PIL.Image.Image], fps: float = 60.):
        return clipify_rgb(frames, fps=fps)


class FlowVisualiser(Visualiser):
    def _clipify_frames(self, frames: List[PIL.Image.Image], fps: float = 30):
        return clipify_flow(frames, fps=fps)
