import pandas as pd
from ast import literal_eval
from typing import Union, Optional

from epic_kitchens._utils import maybe_download
from . import _urls, get_datadir

_training_labels = None  # type: pd.DataFrame
_training_object_labels = None  # type: pd.DataFrame


def _initialise_training_labels() -> None:
    global _training_labels
    labels_path = get_datadir() / "EPIC_train_action_labels.pkl"
    maybe_download(_urls.training_labels_url, labels_path)
    if _training_labels is None:
        _training_labels = pd.read_pickle(labels_path)


def _initialise_training_object_labels() -> None:
    global _training_object_labels
    labels_path = get_datadir() / "EPIC_train_object_labels.csv"
    maybe_download(_urls.training_object_labels_url, labels_path)
    if _training_object_labels is None:
        _training_object_labels = pd.read_csv(
            labels_path, converters={"bounding_boxes": literal_eval}
        )


def training_labels() -> pd.DataFrame:
    _initialise_training_labels()
    return _training_labels.copy()


def training_object_labels() -> pd.DataFrame:
    _initialise_training_object_labels()
    return _training_object_labels.copy()
