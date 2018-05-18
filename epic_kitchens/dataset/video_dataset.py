from abc import ABC
from typing import List

import PIL.Image


class VideoSegment(ABC):
    """
    Represents a video segment with an associated label.
    """
    @property
    def id(self):
        raise NotImplementedError()

    @property
    def label(self):
        raise NotImplementedError()

    @property
    def num_frames(self) -> int:
        raise NotImplementedError()


class VideoDataset(ABC):
    """
    A dataset interface for use with :class:`TsnDataset`. Implement this interface if you
    wish to use your dataset with TSN.

    We cannot use torch.utils.data.Dataset because we need to yield information about
    the number of frames per video, which we can't do with the standard
    torch.utils.data.Dataset.
    """
    def __init__(self, class_count):
        self.class_count = class_count

    @property
    def video_segments(self) -> List[VideoSegment]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def load_frames(self, segment: VideoSegment, idx: List[int]) -> List[PIL.Image.Image]:
        raise NotImplementedError()

