import pandas as pd
from epic_kitchens.meta import _urls

from pathlib import Path
from typing import Union
from logging import getLogger

from epic_kitchens._utils import maybe_download

_datadir = Path(".")
_LOG = getLogger(__name__)


_video_info = None  # type: pd.DataFrame


def set_datadir(dir_: Union[str, Path]):
    """
    Set directory path to

    Parameters
    ----------
    dir_: Path to directory in which to store all downloaded metadata files

    """
    global _datadir
    _datadir = Path(dir_)
    _LOG.info("Setting data directory to {}".format(_datadir))
    _datadir.mkdir(parents=True, exist_ok=True)


def get_datadir() -> Path:
    return _datadir


def video_info():
    global _video_info
    if _video_info is None:
        file_path = get_datadir() / "EPIC_video_info.csv"
        maybe_download(_urls.video_info_url, file_path)
        _video_info = pd.read_csv(file_path, index_col="video")
        _video_info.index.name = "video_id"
    return _video_info.copy()
