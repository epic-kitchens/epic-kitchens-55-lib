import glob
import os
from typing import Dict, Any, List, Iterator

import pandas as pd
from gulpio.adapters import AbstractDatasetAdapter
from gulpio.utils import find_images_in_folder, resize_images

from epic_kitchens.labels import (
    VERB_CLASS_COL,
    VERB_COL,
    VIDEO_ID_COL,
    UID_COL,
    NARRATION_COL,
    PARTICIPANT_ID_COL,
)

Result = Dict[str, Any]


class EpicDatasetAdapter(AbstractDatasetAdapter):
    """Gulp Dataset Adapter for Gulping RGB frames extracted from the EPIC-KITCHENS dataset
    """

    def __init__(
        self,
        video_segment_dir: str,
        annotations_df: pd.DataFrame,
        frame_size: int = -1,
        extension: str = "jpg",
        labelled: bool = True,
    ) -> None:
        """ Gulp all action segments in  ``annotations_df`` reading the dumped frames from
        ``video_segment_dir``

        Args:
            video_segment_dir:
                Root directory containing segmented frames::

                    frame-segments/
                    ├── P01
                    │   ├── P01_01
                    │   |   ├── P01_01_0_open-door
                    │   |   |   ├── frame_0000000008.jpg
                    │   |   |   ...
                    │   |   |   ├── frame_0000000202.jpg
                    │   |   ...
                    │   |   ├── P01_01_329_put-down-plate
                    │   |   |   ├── frame_0000098424.jpg
                    │   |   |   ...
                    │   |   |   ├── frame_0000098501.jpg
                    │   ...

            annotations_df:
                DataFrame containing labels to be gulped.
            frame_size:
                Size of shortest edge of the frame, if not already this size then it will
                be resized.
            extension:
                Extension of dumped frames.
        """
        self.video_segment_dir = video_segment_dir
        self.frame_size = int(frame_size)
        self.meta_data = self._transform_annotations(annotations_df, labelled)
        self.extensions = {"jpg", "jpeg", extension}

    def iter_data(self, slice_element=None) -> Iterator[Result]:
        """Get frames and metadata corresponding to segment

        Args:
            slice_element (optional): If not specified all frames for the segment will be returned

        Yields:
            dict: dictionary with the fields

            * ``meta``: All metadata corresponding to the segment, this is the same as the data
              in the labels csv
            * ``frames``: list of :class:`PIL.Image.Image` corresponding to the frames specified
              in ``slice_element``
            * ``id``: UID corresponding to segment
        """
        slice_element = slice_element or slice(0, len(self))
        for meta in self.meta_data[slice_element]:
            clip_id = self._segment_metadata_to_clip_id(meta)
            folder = os.path.join(
                self.video_segment_dir,
                meta[PARTICIPANT_ID_COL],
                meta[VIDEO_ID_COL],
                clip_id,
            )
            paths = self._find_frames(folder)
            frames = list(resize_images(paths, self.frame_size))
            meta["frame_size"] = frames[0].shape
            meta["num_frames"] = len(frames)
            result = {"meta": meta, "frames": frames, "id": meta[UID_COL]}
            yield result

    def __len__(self):
        return len(self.meta_data)

    def _transform_annotations(
        self, annotations: pd.DataFrame, labelled: bool
    ) -> List[Dict]:
        data = []
        for i, row in annotations.iterrows():
            if labelled:
                assert not pd.isnull(
                    row[VERB_CLASS_COL]
                ), "Row at index has no verb cluster".format(i)
                assert (
                    not len(row[VERB_COL]) == 0
                ), "Row at index {} has empty verb".format(i)
            metadata = row.to_dict()
            if NARRATION_COL not in metadata:
                metadata[NARRATION_COL] = "unnarrated"
            metadata["uid"] = i
            data.append(metadata)
        return data

    def _segment_metadata_to_clip_id(self, meta):
        clip_id = "{video_id}_{uid}_{narration}".format(
            video_id=meta[VIDEO_ID_COL],
            uid=meta[UID_COL],
            narration=meta[NARRATION_COL].lower().strip().replace(" ", "-"),
        )
        return clip_id

    def _find_frames(self, folder):
        frame_paths = find_images_in_folder(folder, formats=self.extensions)
        if not len(frame_paths) > 0:
            raise MissingDataException("{} is not present".format(folder))
        return frame_paths


class EpicFlowDatasetAdapter(EpicDatasetAdapter):
    """Gulp Dataset Adapter for Gulping flow frames extracted from the EPIC-KITCHENS dataset"""

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for meta in self.meta_data[slice_element]:
            paths = self._find_uv_frames(meta)

            frames = {}
            for axis in "u", "v":
                frames[axis] = list(resize_images(paths[axis], self.frame_size))

            meta["frame_size"] = frames["u"][0].shape
            meta["num_frames"] = len(frames["u"])
            result = {
                "meta": meta,
                "frames": list(_intersperse(frames["u"], frames["v"])),
                "id": meta[UID_COL],
            }
            yield result

    def _find_uv_frames(self, meta):
        clip_id = self._segment_metadata_to_clip_id(meta)
        folder = {}
        paths = {}
        for axis in "u", "v":
            folder[axis] = os.path.join(
                self.video_segment_dir,
                meta[PARTICIPANT_ID_COL],
                meta[VIDEO_ID_COL],
                axis,
                clip_id,
            )
            paths[axis] = find_images_in_folder(folder[axis], formats=self.extensions)
            if not len(paths[axis]) > 0:
                raise MissingDataException("{} is not present".format(folder["u"]))
        return paths


def _intersperse(*lists):
    """
    Args:
        *lists:

    Examples:
        >>> list(_intersperse(['a', 'b']))
        ['a', 'b']
        >>> list(_intersperse(['a', 'c'], ['b', 'd']))
        ['a', 'b', 'c', 'd']
        >>> list(_intersperse(['a', 'd'], ['b', 'e'], ['c', 'f']))
        ['a', 'b', 'c', 'd', 'e', 'f']
        >>> list(_intersperse(['a', 'd', 'g'], ['b', 'e'], ['c', 'f']))
        ['a', 'b', 'c', 'd', 'e', 'f']

    """
    i = 0
    min_length = min(map(len, lists))
    total_element_count = len(lists) * min_length
    for i in range(0, total_element_count):
        list_index = i % len(lists)
        element_index = i // len(lists)
        yield lists[list_index][element_index]


class MissingDataException(Exception):
    pass
