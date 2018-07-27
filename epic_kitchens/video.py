import logging
import os
import re
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Tuple, Iterator, List, Iterable

from epic_kitchens.labels import NARRATION_COL, START_TS_COL, STOP_TS_COL, VIDEO_ID_COL
from epic_kitchens.time import timestamp_to_frame

LOG = logging.getLogger(__name__)

Timestamp = str


class Modality(ABC):
    """Interface that a modality extracted from video must implement"""

    def frame_iterator(self, start: Timestamp, stop: Timestamp) -> Iterable[int]:
        """
        Args:
            start: start time (timestamp: HH:MM:SS)
            stop: stop time (timestamp: HH:MM:SS)

        Yields:
            frame indices corresponding to segment from start time to stop time
        """
        raise NotImplementedError


class RGBModality(Modality):
    def __init__(self, fps):
        self.fps = fps

    def frame_iterator(self, start: Timestamp, stop: Timestamp) -> Iterable[int]:
        start_frame = timestamp_to_frame(start, self.fps)
        stop_frame = timestamp_to_frame(stop, self.fps)
        return range(start_frame, stop_frame)


class FlowModality(Modality):
    def __init__(self, dilation=1, stride=1, bound=20, rgb_fps=59.94):
        self.dilation = dilation
        self.stride = stride
        self.bound = bound
        self.rgb_fps = rgb_fps

    def frame_iterator(self, start: Timestamp, stop: Timestamp) -> Iterable[int]:
        start_frame = self._seconds_to_flow_frame_index(start)
        stop_frame = self._seconds_to_flow_frame_index(stop)
        return range(start_frame, stop_frame)

    def _seconds_to_flow_frame_index(self, timestamp: str):
        rgb_frame_index = timestamp_to_frame(timestamp, self.rgb_fps)
        return int(np.ceil(rgb_frame_index / self.stride))


def iterate_frame_dir(root: Path) -> Iterator[Tuple[Path, Path]]:
    """ Iterate over a directory of video dirs with the hierarchy ``root/P01/P01_01/``

    Args:
        root: Root directory with person directory children, then each person directory has
              video directory children e.g. root -> P01 -> P01_01
    Yields:
        (person_dir, video_dir)
    """
    for person_dir in root.iterdir():
        for video_dir in person_dir.iterdir():
            yield (person_dir, video_dir)


def split_dataset_frames(modality: Modality, frames_dir: Path, segment_root_dir: Path,
                         annotations: pd.DataFrame, frame_format='frame%06d.jpg', pattern=re.compile('.*')):
    assert frames_dir.exists()

    frames_dir = frames_dir.resolve()
    segment_root_dir.mkdir(exist_ok=True, parents=True)
    segment_root_dir = segment_root_dir.resolve()

    for person_dir, video_dir in iterate_frame_dir(frames_dir):
        if pattern.search(str(video_dir)):
            annotations_for_video = annotations[annotations[VIDEO_ID_COL] == video_dir.name]
            split_video_frames(modality, frame_format, annotations_for_video, segment_root_dir, video_dir)


def split_video_frames(modality: Modality, frame_format: str, video_annotations: pd.DataFrame,
                       segment_root_dir: Path, video_dir: Path):
    for annotation in video_annotations.itertuples():
        segment_dir_name = "{video_id}_{index}_{narration}".format(
                index=annotation.Index,
                video_id=getattr(annotation, VIDEO_ID_COL),
                narration=getattr(annotation, NARRATION_COL).strip().lower().replace(' ', '-')
        )
        segment_dir = segment_root_dir.joinpath(segment_dir_name)
        segment_dir.mkdir(parents=True, exist_ok=True)
        start_timestamp = getattr(annotation, START_TS_COL)
        stop_timestamp = getattr(annotation, STOP_TS_COL)
        frame_iterator = modality.frame_iterator(start_timestamp, stop_timestamp)

        LOG.info('Linking {video_id} - {narration} - {start}--{stop}'.format(
                video_id=getattr(annotation, VIDEO_ID_COL),
                narration=getattr(annotation, NARRATION_COL),
                start=start_timestamp,
                stop=stop_timestamp
        ))
        _split_frames_by_segment(frame_format, frame_iterator, segment_dir, video_dir)


def _split_frames_by_segment(frame_format: str, frame_iterator, segment_dir: Path, video_dir: Path):
    segment_dir_fd = os.open(str(segment_dir), os.O_RDONLY)
    try:
        first_frame_index = -1
        last_frame_index = -1
        for frame_index in frame_iterator:
            if first_frame_index == -1:
                first_frame_index = frame_index
            source_frame_filename = frame_format % frame_index
            target_frame_filename = frame_format % (frame_index - first_frame_index + 1)
            source_frame_path = video_dir.joinpath(source_frame_filename)
            segmented_frame_path = segment_dir.joinpath(target_frame_filename)
            assert source_frame_path.exists(), "{source_frame_path} does not exist".format(
                source_frame_path=source_frame_path
            )
            if os.path.lexists(str(segmented_frame_path)):
                os.remove(str(segmented_frame_path))
            source_frame_relative_path = os.path.relpath(str(source_frame_path), start=str(segment_dir))
            os.symlink(str(source_frame_relative_path), str(segmented_frame_path), dir_fd=segment_dir_fd)
            last_frame_index = frame_index
        LOG.info('  Linked [{first},{last}] -> [1, {last_target}]'.format(
                first=first_frame_index,
                last=last_frame_index,
                last_target=last_frame_index - first_frame_index + 1
        ))
    finally:
        os.close(segment_dir_fd)
