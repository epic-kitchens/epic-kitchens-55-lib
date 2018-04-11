import glob
import os
from typing import Dict, Any, List, Iterator

import pandas as pd
from gulpio.adapters import AbstractDatasetAdapter
from gulpio.utils import find_images_in_folder, resize_images

from epic_kitchens.labels import VERB_CLASS_COL, VERB_COL, VIDEO_ID_COL, UID_COL, NARRATION_COL, \
    PARTICIPANT_ID_COL

Result = Dict[str, Any]


class EpicDatasetAdapter(AbstractDatasetAdapter):
    """Gulp Dataset Adapter for Gulping RGB frames extracted from the EPIC-KITCHENS dataset
    """

    def __init__(self, video_segment_dir: str, annotations_df: pd.DataFrame, frame_size=-1,
                 extension='jpg') -> None:
        """ Gulp all action segments in  ``annotations_df`` reading the dumped frames from
        ``video_segment_dir``

        Args:
            video_segment_dir: Root directory containing segmented frames::

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

            annotations_df: DataFrame containing labels to be gulped.
            frame_size (optional): Size of shortest edge of the frame, if not already this size then it will
                be resized.
            extension (optional): Extension of dumped frames.
        """
        self.video_segment_dir = video_segment_dir
        self.frame_size = int(frame_size)
        self.meta_data = self._transform_annotations(annotations_df)
        self.extension = extension

    @staticmethod
    def _transform_annotations(annotations: pd.DataFrame) -> List[Dict]:
        data = []
        for i, row in annotations.iterrows():
            assert not pd.isnull(row[VERB_CLASS_COL]), "Row at index has no verb cluster".format(i)
            assert not len(row[VERB_COL]) == 0, "Row at index {} has empty verb".format(i)
            metadata = row.to_dict()
            metadata['uid'] = i
            data.append(metadata)
        return data

    def iter_data(self, slice_element=None) -> Iterator[Result]:
        """Get frames and metadata corresponding to segment

        Args:
            slice_element (optional): If not specified all frames for the segment will be returned

        Returns:
            dictionary with the fields:
            * ``meta``: All metadata corresponding to the segment, this is the same as the data
              in the labels csv
            * ``frames``: list of :py:class:`PIL.Image` corresponding to the frames specified
              in ``slice_element``
            * ``id``: UID corresponding to segment
        """
        slice_element = slice_element or slice(0, len(self))
        for meta in self.meta_data[slice_element]:
            clip_id = "{video_id}_{uid}_{narration}".format(
                    video_id=meta[VIDEO_ID_COL],
                    uid=meta[UID_COL],
                    narration=meta[NARRATION_COL].lower().strip().replace(' ', '-'))
            folder = os.path.join(self.video_segment_dir,
                                  meta[PARTICIPANT_ID_COL],
                                  meta[VIDEO_ID_COL],
                                  clip_id)
            frame_paths = find_images_in_folder(folder, formats=['jpg', 'jpeg'])
            frames = list(resize_images(frame_paths, self.frame_size))
            if len(frames) > 0:
                meta['frame_size'] = frames[0].shape
                meta['num_frames'] = len(frames)
                result = {'meta': meta,
                          'frames': frames,
                          'id': meta[UID_COL]}
                yield result
            else:
                raise MissingDataException("{} is not present".format(folder))

    def __len__(self):
        return len(self.meta_data)


class EpicFlowDatasetAdapter(EpicDatasetAdapter):
    """Gulp Dataset Adapter for Gulping flow frames extracted from the EPIC-KITCHENS dataset
    """

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for meta in self.meta_data[slice_element]:
            clip_id = "{video_id}_{uid}_{narration}".format(
                    video_id=meta[VIDEO_ID_COL],
                    uid=meta[UID_COL],
                    narration=meta[NARRATION_COL].lower().strip().replace(' ', '-'))
            folder = {}
            paths = {}
            frames = {}
            for axis in 'u', 'v':
                folder[axis] = os.path.join(self.video_segment_dir, meta[PARTICIPANT_ID_COL],
                                            meta[VIDEO_ID_COL],
                                            axis, clip_id)
                paths[axis] = glob.glob(folder[axis] + os.path.sep + '*.' + self.extension, recursive=True)
                frames[axis] = list(resize_images(paths[axis], self.frame_size))
            if len(frames['u']) > 0:
                meta['frame_size'] = frames['u'][0].shape
                meta['num_frames'] = len(frames['u'])
                result = {'meta': meta,
                          'frames': list(_intersperse(frames['u'], frames['v'])),
                          'id': meta[UID_COL]}
                yield result
            else:
                raise MissingDataException("{} is not present".format(folder['u']))

    def __len__(self):
        return len(self.meta_data)


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