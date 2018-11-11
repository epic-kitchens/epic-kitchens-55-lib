import pandas as pd
from typing import Union

from .._utils import maybe_download
from . import get_datadir, _urls

_narrations = None  # type: pd.DataFrame


def training_narrations() -> pd.DataFrame:
    global _narrations
    if _narrations is None:
        file_path = get_datadir() / "EPIC_train_action_narrations.csv"
        maybe_download(_urls.training_narrations_url, file_path)
        _narrations = pd.read_csv(file_path)
    return _narrations
