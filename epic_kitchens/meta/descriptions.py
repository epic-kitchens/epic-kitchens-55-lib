import pandas as pd
from typing import Union

from epic_kitchens._utils import maybe_download
from epic_kitchens.meta import get_datadir, _urls

_descriptions = None  # type: pd.DataFrame


def descriptions() -> pd.DataFrame:
    global _descriptions
    if _descriptions is None:
        file_path = get_datadir() / "EPIC_descriptions.csv"
        maybe_download(_urls.descriptions_url, file_path)
        _descriptions = pd.read_csv(
            file_path, index_col="video_id", parse_dates={"datetime": ["date", "time"]}
        )
    return _descriptions.copy()
