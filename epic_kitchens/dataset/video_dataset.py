from abc import ABC
from typing import List, Optional, Callable

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

    We cannot use :class:`torch.utils.data.Dataset` because we need to yield information about
    the number of frames per video, which we can't do with the standard
    torch.utils.data.Dataset.
    """

    def __init__(
        self,
        class_count,
        segment_filter: Optional[Callable[[VideoSegment], bool]] = None,
        sample_transform: Optional[
            Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]]
        ] = None,
    ) -> None:
        self.class_count = class_count
        if segment_filter is None:
            self.segment_filter = lambda _: True
        else:
            self.segment_filter = segment_filter
        if sample_transform is None:
            self.sample_transform = lambda x: x
        else:
            self.sample_transform = sample_transform

    @property
    def video_segments(self) -> List[VideoSegment]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def load_frames(
        self, segment: VideoSegment, idx: Optional[List[int]] = None
    ) -> List[PIL.Image.Image]:
        raise NotImplementedError()
