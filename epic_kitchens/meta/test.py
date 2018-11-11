import pandas as pd
from typing import Union

from .._utils import maybe_download
from . import _urls
from . import get_datadir

_test_seen_timestamps = None  # type: pd.DataFrame
_test_unseen_timestamps = None  # type: pd.DataFrame


def _initialise_test_seen_timestamps() -> None:
    global _test_seen_timestamps
    timestamps_path = get_datadir() / "EPIC_test_seen_timestamps.pkl"
    maybe_download(_urls.test_seen_timestamps_url, timestamps_path)
    if _test_seen_timestamps is None:
        _test_seen_timestamps = pd.read_pickle(timestamps_path)
        _test_seen_timestamps["split"] = "seen"


def _initialise_test_unseen_timestamps() -> None:
    global _test_unseen_timestamps
    timestamps_path = get_datadir() / "EPIC_test_unseen_timestamps.pkl"
    maybe_download(_urls.test_unseen_timestamps_url, timestamps_path)
    if _test_unseen_timestamps is None:
        _test_unseen_timestamps = pd.read_pickle(timestamps_path)
        _test_unseen_timestamps["split"] = "unseen"


def test_timestamps(split: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    split: 'seen', 'unseen', or 'all' (loads both with a 'split'

    Returns
    -------
    Timestamps as a ``pd.DataFrame`` containing the rows
      - uid: int, Unique identified of the action segment.
      - participant_id: str (e.g. 'P01'), ID of the participant
      - video_id: str (e.g. 'P01_11'), Video the action segment belongs to
      - start_timestamp: str (e.g. '00:00:00.000'), Start time in 'HH:mm:ss.SSS' of the action
      - stop_timestamp: str (e.g. '00:01:03.890'), Stop time in 'HH:mm:ss.SSS' of the action
      - start_frame: int (e.g. 1), Start frame of the action (for frames provided by EPIC-Kitchens)
      - stop_frame: int (e.g. 203) Stop frame of the action (for frames provided by EPIC-Kitchens)

    """
    split = split.strip().lower()
    if split == "all":
        _initialise_test_seen_timestamps()
        _initialise_test_unseen_timestamps()
        return pd.concat([_test_seen_timestamps, _test_unseen_timestamps])

    if split == "seen":
        _initialise_test_seen_timestamps()
        return _test_seen_timestamps.copy()
    elif split == "unseen":
        _initialise_test_unseen_timestamps()
        return _test_unseen_timestamps.copy()
    raise ValueError("Unknown split '{}' expected 'seen' or 'unseen'".format(split))
