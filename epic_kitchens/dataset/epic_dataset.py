import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Iterable

import PIL.Image
from gulpio import GulpDirectory

from epic_kitchens.labels import VERB_CLASS_COL, NOUN_CLASS_COL, UID_COL
from epic_kitchens.dataset.video_dataset import VideoDataset, VideoSegment


SegmentFilter = Callable[[VideoSegment], bool]
ClassGetter = Callable[[Dict[str, Any]], Any]
VideoTransform = Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]]


def _verb_class_getter(metadata):
    return int(metadata[VERB_CLASS_COL])


def _noun_class_getter(metadata):
    return int(metadata[NOUN_CLASS_COL])


_class_getters = {
    "verb": _verb_class_getter,
    "noun": _noun_class_getter,
    "verb+noun": lambda metadata: {
        "verb": _verb_class_getter(metadata),
        "noun": _noun_class_getter(metadata),
    },
    None: lambda meta: None,
}

_verb_class_count = 125
_noun_class_count = 353
_class_count = {
    "verb": _verb_class_count,
    "noun": _noun_class_count,
    "verb+noun": (_verb_class_count, _noun_class_count),
    None: 0,
}


class GulpVideoSegment(VideoSegment):
    """SegmentRecord for a video segment stored in a gulp file.

    Assumes that the video segment has the following metadata in the gulp file:
      - id
      - num_frames
    """

    def __init__(
        self,
        gulp_metadata_dict: Dict[str, Any],
        class_getter: Callable[[Dict[str, Any]], Any],
    ) -> None:
        self.metadata = gulp_metadata_dict
        self.class_getter = class_getter
        self.gulp_index = gulp_metadata_dict[UID_COL]

    @property
    def id(self) -> str:
        """ID of video segment"""
        return self.gulp_index

    @property
    def label(self) -> Any:
        cls = self.class_getter(self.metadata)
        # WARNING: this type check should be removed once we regulp our data
        # so that classes are ints in the metadata json
        if isinstance(cls, float):
            return int(cls)
        else:
            return cls

    @property
    def num_frames(self) -> int:
        """Number of video frames"""
        return self.metadata["num_frames"]

    def __getitem__(self, item):
        return self.metadata[item]

    def __getattr__(self, item):
        return self.metadata[item]

    def __str__(self):
        return "GulpVideoSegment[label={label}, num_frames={num_frames}]".format(
            label=self.label, num_frames=self.num_frames
        )

    def __repr__(self):
        return "GulpVideoSegment({metadata}, {class_getter})".format(
            metadata=repr(self.metadata), class_getter=repr(self.class_getter)
        )


class EpicVideoDataset(VideoDataset):
    """VideoDataset for gulped RGB frames"""

    def __init__(
        self,
        gulp_path: Union[Path, str],
        class_type: str,
        *,
        with_metadata: bool = False,
        class_getter: Optional[ClassGetter] = None,
        segment_filter: Optional[SegmentFilter] = None,
        sample_transform: Optional[VideoTransform] = None
    ) -> None:
        """
        Args:
            gulp_path: Path to gulp directory containing the gulped EPIC RGB or flow frames

            class_type: One of verb, noun, verb+noun, None, determines what label the segment
                returns. ``None`` should be used for loading test datasets.

            with_metadata: When True the segments will yield a tuple (metadata, class) where the
                class is defined by the class getter and the metadata is the raw dictionary stored
                in the gulp file.

            class_getter: Optionally provide a callable that takes in the gulp dict representing the
                segment from which you should return the class you wish the segment to have.

            segment_filter: Optionally provide a callable that takes a segment and returns True if
                you want to keep the segment in the dataset, or False if you wish to exclude it.

            sample_transform: Optionally provide a sample transform function which takes a list of
                PIL images and transforms each of them. This is applied on the frames just before
                returning from :meth:`load_frames`.
        """
        super().__init__(
            _class_count[class_type],
            segment_filter=segment_filter,
            sample_transform=sample_transform,
        )
        if isinstance(gulp_path, str):
            gulp_path = Path(gulp_path)
        assert gulp_path.exists(), "Could not find the path {}".format(gulp_path)
        self.gulp_dir = GulpDirectory(str(gulp_path))
        if class_getter is None:
            class_getter = _class_getters[class_type]
        if with_metadata:
            original_getter = copy.copy(class_getter)
            class_getter = lambda metadata: (metadata, original_getter(metadata))
        self._video_segments = self._read_segments(
            self.gulp_dir.merged_meta_dict, class_getter
        )

    @property
    def video_segments(self) -> List[VideoSegment]:
        """
        List of video segments that are present in the dataset. The describe the start and stop
        times of the clip and its class.
        """
        return list(self._video_segments.values())

    def load_frames(
        self, segment: VideoSegment, indices: Optional[Iterable[int]] = None
    ) -> List[PIL.Image.Image]:
        """
        Load frame(s) from gulp directory.

        Args:
            segment: Video segment to load
            indices: Frames indices to read

        Returns:
            Frames indexed by ``indices`` from the ``segment``.

        """
        if indices is None:
            indices = range(0, segment.num_frames)
        selected_frames = []  # type: List[PIL.Image.Image]
        for i in indices:
            # Without passing a slice to the gulp directory index we load ALL the frames
            # so we create a slice with a single element -- that way we only read a single frame
            # from the gulp chunk, and not the whole chunk.
            # Here we also apply the sample transform to the loaded frames
            frames = self._sample_video_at_index(segment, i)
            frames = self.sample_transform(frames)
            selected_frames.extend(frames)
        return selected_frames

    def __len__(self):
        return len(self.video_segments)

    def __getitem__(self, id):
        return self._video_segments[id]

    def _read_segments(
        self, gulp_dir_meta_dict, class_getter: Callable[[Dict[str, Any]], Any]
    ) -> Dict[str, VideoSegment]:
        segments = dict()  # type: Dict[str, VideoSegment]
        for video_id in gulp_dir_meta_dict:
            segment = GulpVideoSegment(
                gulp_dir_meta_dict[video_id]["meta_data"][0], class_getter
            )
            if self.segment_filter(segment):
                segments[segment.id] = segment
        return segments

    def _sample_video_at_index(
        self, record: VideoSegment, index: int
    ) -> List[PIL.Image.Image]:
        single_frame_slice = slice(index, index + 1)
        numpy_frame = self.gulp_dir[record.id, single_frame_slice][0][0]
        return [PIL.Image.fromarray(numpy_frame).convert("RGB")]


class EpicVideoFlowDataset(EpicVideoDataset):
    """VideoDataset for loading gulped flow. The loader assumes that flow :math:`u`, :math:`v`
    frames are stored alternately in a flat manner: :math:`[u_0, v_0, u_1, v_1, \ldots, u_n, v_n]`

    """

    def _sample_video_at_index(
        self, record: VideoSegment, index: int
    ) -> List[PIL.Image.Image]:
        # Flow pairs are stored in a contiguous manner in the gulp chunk:
        # [u_1, v_1, u_2, v_2, ..., u_n, v_n]
        # so we have to convert our desired frame index i to the gulp
        # indices j by j = (i * 2, (i + 1) * 2)
        flow_pair_slice = slice(index * 2, (index + 1) * 2)
        numpy_frames = self.gulp_dir[record.id, flow_pair_slice][0]
        frames = [
            PIL.Image.fromarray(numpy_frame).convert("L")
            for numpy_frame in numpy_frames
        ]
        return frames
