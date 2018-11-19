import logging
import shutil

import re
import ujson as json

import glob
from pathlib import Path

from typing import Dict, Any, List, Union, Iterator, Callable, Pattern

from epic_kitchens.internal.utils import numpy_to_builtin

_LOG = logging.getLogger(__name__)


GulpSegmentId = str
GulpMetaDataDict = Dict[GulpSegmentId, Any]
GulpFrameInfo = List[List[int]]
GulpSegmentMetaDict = Dict[str, Union[GulpFrameInfo, List[GulpMetaDataDict]]]
GulpDirectoryMetaDict = Dict[GulpSegmentId, GulpSegmentMetaDict]


def modify_all_gulp_dirs(
    gulp_dir_root: Path,
    transform_func: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    gulp_dir_pattern: Pattern = re.compile(".*gulped.*"),
    drop_nones: bool = False,
    skip_backup: bool = False,
):
    """

    Args:
        gulp_dir_root: A directory in which there are multiple subdirectories, each of which is a gulp
            directory
        transform_func: A callable that takes ``(id, meta_data)`` and returns ``meta_data``
            transformed in some way.
        gulp_dir_pattern: Regexp for matching gulp directories within ``gulp_dir_root``
        drop_nones: If ``transform_func`` returns None and this is True, the segment will be dropped from
            the gulp directory
        skip_backup: Skip making ``.bak`` files of the .gmeta files
    """
    gulp_dirs = [
        child_dir
        for child_dir in gulp_dir_root.iterdir()
        if gulp_dir_pattern.search(child_dir.name)
    ]
    for gulp_dir in gulp_dirs:
        modify_metadata(
            gulp_dir, transform_func, skip_backup=skip_backup, drop_nones=drop_nones
        )


def modify_metadata(
    gulp_dir: Path,
    transform_func: Callable[
        [GulpSegmentId, GulpMetaDataDict], Union[GulpMetaDataDict, None]
    ],
    *,
    drop_nones=False,
    skip_backup=False
) -> None:
    """Modify metadata in a gulp directory (and backup existing metadata files with .bak suffix) by
    providing transform function

    Args:
        gulp_dir: Gulp directory containing .gmeta and .gulp files
        transform_func: User provided function to transform the metadata of a gulp segment in some way
        drop_nones: If set and ``transform_func`` returns None, then remove the segment from the gulp meta dict
        skip_backup: Skip making ``.bak`` files for all ``.gmeta`` files
    """
    meta_dicts = dict()
    for meta_path in _find_meta_files(gulp_dir):
        with meta_path.open(mode="r", encoding="utf-8") as f:
            meta_dicts[meta_path] = json.load(f)

    _LOG.info("Modifying {}".format(gulp_dir))
    for meta_path, meta_dict in meta_dicts.items():
        segment_ids = set(meta_dict.keys())
        for segment_id in segment_ids:
            _update_metadata(
                segment_id, meta_dict, transform_func, drop_nones=drop_nones
            )

        if not skip_backup:
            _backup_meta_data(meta_path)

        with meta_path.open(mode="w", encoding="utf-8") as f:
            json.dump(meta_dict, f)


def iterate_gulp_dir_metadata_files(gulp_dir: Path) -> Iterator[Path]:
    """Iterate over gulp directory yielding metadata file paths

    Args:
        gulp_dir: Directory containing *.gmeta and *.gulp files.

    Yields:
        Path of a gmeta file
    """
    metadata_pattern = re.compile(r"^meta_\d+\.gmeta")
    for child in gulp_dir.iterdir():
        if metadata_pattern.match(child.name):
            yield child


def iterate_metadata(
    full_meta_dict: GulpDirectoryMetaDict
) -> Iterator[GulpMetaDataDict]:
    """
    Args:
        full_meta_dict: Full metadata dict (e.g. from :method:`GulpDirectory.merged_meta_dict`

    Yields:
        meta_data dictionary for a single segment.
    """
    for id_ in full_meta_dict.keys():
        yield full_meta_dict[id_]["meta_data"][0]


def _find_meta_files(path: Path) -> List[Path]:
    return sorted(list(map(Path, glob.glob(str(path.joinpath("meta*.gmeta"))))))


def _backup_meta_data(meta_path: Path) -> None:
    meta_path = meta_path.resolve()
    backup_meta_path = meta_path.parent / (meta_path.name + ".bak")
    i = 0
    while backup_meta_path.exists():
        backup_meta_path = backup_meta_path.with_suffix(".bak{}".format(i))
        i += 1
    shutil.copy(str(meta_path), str(backup_meta_path))


def _update_metadata(segment_id, meta_dict, transform_func, *, drop_nones=False):
    segment_metadata = meta_dict[segment_id]["meta_data"]
    if isinstance(segment_metadata, list):
        segment_metadata = segment_metadata[0]
    # We have to convert numpy values to python values as the json module can't dump them
    transformed_metadata = transform_func(segment_id, segment_metadata)
    if transformed_metadata is None and drop_nones:
        _LOG.info("Dropping {} since transform_func returned None".format(segment_id))
        del meta_dict[segment_id]
    else:
        meta_dict[segment_id]["meta_data"] = [numpy_to_builtin(transformed_metadata)]
