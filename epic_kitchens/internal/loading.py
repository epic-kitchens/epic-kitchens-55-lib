"""
Loading annotations from the web is essentially the same problem regardless of what we're
loading. We want to download if the file doesn't exist, and then load it if it hasn't already been loaded
on top of this we want to version the files so we can load in different label versions in the same
session, e.g. for comparing what has changed between versions.

The process is the same:
- If the file isn't downloaded, download it to a version subfolder
- Read the file if it hasn't already been read, into an namespace divided by versions
- Potentially we also want to produce multiple data structures from each file, I'm unsure whether
  this belongs as part of this process or whether it should be separate. Consider the verb classes,
  we want to dataframe, but we also want to build an inverse lookup table from verbs to verb class.
  We could yield multiple (name, data_structure) pairs from the loading functions, but then accessing data
  structures isn't based on the filename alone, which seems a bit nasty as you have to know the name of
  the data structures returned by the loading functions to access them.

We might want to split this into a two stage process where we have this web-backed file repository
that downloads and caches the data and then another object responsible for then deriving data structures
from those files.
I'm not sure what the interface of the second object would look like though
"""
import os
import urllib.request
from ast import literal_eval

import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any, Dict
import logging

_LOG = logging.getLogger(__name__)
Loader = Callable[[Path], Any]


def _read_video_info(fp):
    df = pd.read_csv(fp, index_col="video")
    df.index.name = "video_id"
    return df


_file_loaders = {
    "EPIC_verb_classes.csv": lambda fp: pd.read_csv(
        fp, index_col="verb_id", converters={"verbs": literal_eval}
    ),
    "EPIC_noun_classes.csv": lambda fp: pd.read_csv(
        fp, index_col="noun_id", converters={"nouns": literal_eval}
    ),
    "EPIC_many_shot_actions.csv": lambda fp: set(
        pd.read_csv(
            fp, index_col="action_class", converters={"action_class": literal_eval}
        ).index.values
    ),
    "EPIC_many_shot_verbs.csv": lambda fp: set(
        pd.read_csv(fp, index_col="verb_class").index.values
    ),
    "EPIC_many_shot_nouns.csv": lambda fp: set(
        pd.read_csv(fp, index_col="noun_class").index.values
    ),
    "EPIC_descriptions.csv": lambda fp: pd.read_csv(
        fp, index_col="video_id", parse_dates={"datetime": ["date", "time"]}
    ),
    "EPIC_video_info.csv": _read_video_info,
    "EPIC_train_action_narrations.csv": lambda fp: pd.read_csv(fp),
    "EPIC_train_object_labels.csv": lambda fp: pd.read_csv(
        fp, converters={"bounding_boxes": literal_eval}
    ),
}

_filetype_loaders = {"csv": pd.read_csv, "pkl": pd.read_pickle}


class AnnotationRepository:
    def __init__(
        self, version: str = "v1.5.0", local_dir: Optional[Path] = None
    ) -> None:
        self.version = version
        base_url = "https://github.com/epic-kitchens/annotations/raw/{}/".format(
            version
        )
        if local_dir is None:
            cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home()))
            local_dir = cache_dir / "epic_kitchens" / version
        # local_dir.mkdir(exist_ok=True, parents=True)
        self.http_folder = HttpFolder(
            base_url, local_dir, Loaders(_file_loaders, _filetype_loaders)
        )
        self.inverse_lookups = {"verb": None, "noun": None}

    @property
    def local_dir(self):
        return self.http_folder.local_dir

    @local_dir.setter
    def local_dir(self, dir_: Path):
        self.http_folder.local_dir = dir_

    def inverse_noun_lookup(self):
        if self.inverse_lookups["noun"] is None:
            inverse_lookup = dict()
            for noun_class, row in self.noun_classes().iterrows():
                for noun in row["nouns"]:
                    inverse_lookup[noun] = noun_class
            self.inverse_lookups["noun"] = inverse_lookup
        return self.inverse_lookups["noun"]

    def inverse_verb_lookup(self):
        if self.inverse_lookups["verb"] is None:
            inverse_lookup = dict()
            for verb_class, row in self.verb_classes().iterrows():
                for verb in row["verbs"]:
                    inverse_lookup[verb] = verb_class
            self.inverse_lookups["verb"] = inverse_lookup
        return self.inverse_lookups["verb"]

    def train_action_labels(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_train_action_labels.pkl")

    def train_object_labels(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_train_object_labels.csv")

    def test_seen_timestamps(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_test_s1_timestamps.pkl")

    def test_unseen_timestamps(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_test_s2_timestamps.pkl")

    def verb_classes(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_verb_classes.csv")

    def noun_classes(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_noun_classes.csv")

    def many_shot_verbs(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_many_shot_verbs.csv")

    def many_shot_nouns(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_many_shot_nouns.csv")

    def many_shot_actions(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_many_shot_actions.csv")

    def train_action_narrations(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_train_action_narrations.csv")

    def video_descriptions(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_descriptions.csv")

    def video_info(self) -> pd.DataFrame:
        return self.http_folder.load_file("EPIC_video_info.csv")


class Loaders:
    def __init__(
        self, file_loaders: Dict[str, Loader], filetype_loaders: Dict[str, Loader]
    ) -> None:
        self.file_loaders = file_loaders
        self.filetype_loaders = filetype_loaders

    def __getitem__(self, file_name: str):
        file_type = self._get_filetype(file_name)  # drop . from extension
        if file_name in self.file_loaders:
            return self.file_loaders[file_name]
        elif file_type in self.filetype_loaders:
            return self.filetype_loaders[file_type]
        else:
            raise KeyError(
                (
                    "{} not in file_loaders, nor is a default filetype handler registered for {} "
                    "files"
                ).format(file_name, file_type)
            )

    def _get_filetype(self, file_name):
        return os.path.splitext(file_name)[1][1:]


class HttpFolder:
    def __init__(self, base_url: str, local_dir: Path, loaders: Loaders) -> None:
        self.base_url = base_url
        # TODO: Invalidate `files` cache when this is changed
        self.local_dir = local_dir
        self.loaders = loaders
        self.files = dict()  # type: Dict[str, Loader]

    def load_file(self, name: str) -> Any:
        if name not in self.files:
            file_path = self.local_dir / name
            url = self.base_url + name
            self._maybe_download(url, file_path)
            self.files[name] = self.loaders[name](file_path)
        return self.files[name]

    def _maybe_download(self, url: str, file_path: Path) -> None:
        file_path.parent.mkdir(exist_ok=True, parents=True)
        if not file_path.exists():
            urllib.request.urlretrieve(url, str(file_path))
