import copy
from pathlib import Path
from typing import Any, Callable, Dict, List

import PIL.Image
from gulpio import GulpDirectory

from epic_kitchens.labels import VERB_CLASS_COL, NOUN_CLASS_COL, UID_COL
from epic_kitchens.dataset.video_dataset import VideoDataset, VideoSegment


def _verb_class_getter(metadata):
    return int(metadata[VERB_CLASS_COL])


def _noun_class_getter(metadata):
    return int(metadata[NOUN_CLASS_COL])


_class_getter = {
    'verb': _verb_class_getter,
    'noun': _noun_class_getter,
    'verb+noun': lambda metadata: {'verb': _verb_class_getter(metadata),
                                   'noun': _noun_class_getter(metadata)},
}

_verb_class_count = 125
_noun_class_count = 353
_class_count = {
    'verb': _verb_class_count,
    'noun': _noun_class_count,
    'verb+noun': (_verb_class_count, _noun_class_count)
}


class GulpVideoSegment(VideoSegment):
    """ SegmentRecord for a video segment stored in a gulp file.

    Assumes that the video segment has the following metadata in the gulp file:
      - id
      - num_frames
    """

    def __init__(self, gulp_metadata_dict: Dict[str, Any],
                 class_getter: Callable[[Dict[str, Any]], Any]) -> None:
        self.metadata = gulp_metadata_dict
        self.class_getter = class_getter
        self.gulp_index = gulp_metadata_dict[UID_COL]

    @property
    def id(self):
        return self.gulp_index

    @property
    def label(self):
        cls = self.class_getter(self.metadata)
        # WARNING: this type check should be removed once we regulp our data
        # so that classes are ints in the metadata json
        if isinstance(cls, float):
            return int(cls)
        else:
            return cls

    @property
    def num_frames(self) -> int:
        return self.metadata['num_frames']

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
                metadata=repr(self.metadata),
                class_getter=repr(self.class_getter)
        )


class EpicVideoDataset(VideoDataset):
    def __init__(self, gulp_path: Path, class_type: str, class_getter=None, with_metadata=False) -> None:
        super().__init__(_class_count[class_type])
        assert(gulp_path.exists()), "Could not find the path {}".format(gulp_path)
        self.gulp_dir = GulpDirectory(str(gulp_path))
        if class_getter is None:
            class_getter = _class_getter[class_type]
        if with_metadata:
            original_getter = copy.copy(class_getter)
            class_getter = lambda metadata: (metadata, original_getter(metadata))
        self._video_list = self._read_video_records(self.gulp_dir.merged_meta_dict,
                                                    class_getter)

    @property
    def video_segments(self) -> List[VideoSegment]:
        return self._video_list

    def load_frames(self, segment: VideoSegment, indices: List[int]) -> List[PIL.Image.Image]:
        selected_frames = [] # type: List[PIL.Image.Image]
        for i in indices:
            # Without passing a slice to the gulp directory index we load ALL the frames
            # so we create a slice with a single element -- that way we only read a single frame
            # from the gulp chunk, and not the whole chunk.
            frames = self._sample_video_at_index(segment, i)
            selected_frames.extend(frames)
        return selected_frames

    def __len__(self):
        return len(self.video_segments)

    def _read_video_records(self, gulp_dir_meta_dict,
                            class_getter: Callable[[Dict[str, Any]], Any]) -> List[VideoSegment]:
        return [GulpVideoSegment(gulp_dir_meta_dict[video_id]['meta_data'][0],
                                 class_getter)
                for video_id in gulp_dir_meta_dict]

    def _sample_video_at_index(self, record: VideoSegment, index: int) -> List[PIL.Image.Image]:
        single_frame_slice = slice(index, index + 1)
        numpy_frame = self.gulp_dir[record.id, single_frame_slice][0][0]
        return [PIL.Image.fromarray(numpy_frame).convert('RGB')]


class EpicVideoFlowDataset(EpicVideoDataset):
    def _sample_video_at_index(self, record: VideoSegment, index: int) -> List[PIL.Image.Image]:
        # Flow pairs are stored in a contiguous manner in the gulp chunk:
        # [u_1, v_1, u_2, v_2, ..., u_n, v_n]
        # so we have to convert our desired frame index i to the gulp
        # indices j by j = (i * 2, (i + 1) * 2)
        flow_pair_slice = slice(index * 2, (index + 1) * 2)
        numpy_frames = self.gulp_dir[record.id, flow_pair_slice][0]
        frames = [PIL.Image.fromarray(numpy_frame).convert('L') for numpy_frame in numpy_frames]
        return frames
