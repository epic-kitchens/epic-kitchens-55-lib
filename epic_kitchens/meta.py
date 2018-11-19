import pandas as pd
from collections import namedtuple

from epic_kitchens.internal.loading import AnnotationRepository

from pathlib import Path
from typing import Union, Set
from logging import getLogger

_LOG = getLogger(__name__)

_annotation_repositories = {"v1.5.0": AnnotationRepository()}
_annotation_repository = _annotation_repositories["v1.5.0"]

ActionClass = namedtuple("ActionClass", ["verb_class", "noun_class"])
Action = namedtuple("Action", ["verb", "noun"])


def set_version(version: str):
    global _annotation_repository, _annotation_repositories
    if version not in _annotation_repositories:
        _annotation_repositories[version] = AnnotationRepository(
            version=version, local_dir=_annotation_repository.local_dir
        )
    _annotation_repository = _annotation_repositories[version]


def set_datadir(dir_: Union[str, Path]):
    """Set download directory

    Args:
        dir_:
            Path to directory in which to store all downloaded metadata files
    """
    _annotation_repository.local_dir = Path(dir_)
    _LOG.info("Setting data directory to {}".format(dir_))


def get_datadir() -> Path:
    """
    Returns:
        Directory under which any downloaded files are stored, defaults to current working directory
    """
    return _annotation_repository.local_dir


def verb_to_class(verb: str) -> int:
    """
    Args:
        verb:
            A noun from a narration

    Returns:
        The corresponding numeric class of the verb if it exists

    Raises:
        IndexError:
            If the verb doesn't belong to any of the verb classes
    """
    return _annotation_repository.inverse_verb_lookup()[verb]


def noun_to_class(noun: str) -> int:
    """
    Args:
        noun:
            A noun from a narration

    Returns:
        The corresponding numeric class of the noun if it exists

    Raises:
        IndexError:
            If the noun doesn't belong to any of the noun classes
    """
    return _annotation_repository.inverse_noun_lookup()[noun]


def class_to_verb(cls: int) -> str:
    """
    Args:
        cls: numeric verb class

    Returns:
        Canonical verb representing the class

    Raises:
        IndexError:
            if ``cls`` is an invalid verb class
    """
    return _annotation_repository.verb_classes()["class_key"].loc[cls]


def class_to_noun(cls: int) -> str:
    """
    Args:
        cls: numeric noun class

    Returns:
        Canonical noun representing the class

    Raises:
        IndexError:
            if ``cls`` is an invalid noun class

    """
    return _annotation_repository.noun_classes()["class_key"].loc[cls]


def noun_classes() -> pd.DataFrame:
    """
    Get dataframe containing the mapping between numeric noun classes, the canonical noun of that class
    and nouns clustered into the class.

    Returns:
        Dataframe with the columns:

        .. include:: meta/noun_classes.rst
    """
    return _annotation_repository.noun_classes().copy()


def verb_classes() -> pd.DataFrame:
    """
    Get dataframe containing the mapping between numeric verb classes, the canonical verb of that class
    and verbs clustered into the class.

    Returns:
        Dataframe with the columns

        .. include:: meta/verb_classes.rst
    """
    return _annotation_repository.verb_classes().copy()


def many_shot_verbs() -> Set[int]:
    """
    Returns:
        The set of verb classes that are many shot (appear more than 100 times in training).
    """
    return set(_annotation_repository.many_shot_verbs())


def many_shot_nouns() -> Set[int]:
    """
    Returns:
        The set of noun classes that are many shot (appear more than 100 times in training).
    """
    return set(_annotation_repository.many_shot_nouns())


def many_shot_actions() -> Set[ActionClass]:
    """
    Returns:
        The set of actions classes that are many shot (verb_class appears more than 100 times
        in training, noun_class appears more than 100 times in training, and the action appears
        at least once in training).

    """
    return set(_annotation_repository.many_shot_actions())


def is_many_shot_action(action_class: ActionClass) -> bool:
    """
    Args:
        action_class:
            ``(verb_class, noun_class)`` tuple

    Returns:
        Whether action_class is many shot or not
    """
    return action_class in _annotation_repository.many_shot_actions()


def is_many_shot_verb(verb_class: int) -> bool:
    """
    Args:
        verb_class: numeric verb class

    Returns:
        Whether verb_class is many shot or not
    """
    return verb_class in _annotation_repository.many_shot_verbs()


def is_many_shot_noun(noun_class: int) -> bool:
    """
    Args:
        noun_class: numeric noun class

    Returns:
        Whether noun class is many shot or not

    """
    return noun_class in _annotation_repository.many_shot_nouns()


def training_narrations() -> pd.DataFrame:
    """
    Returns:
        Dataframe with the columns

        .. include:: meta/train_action_narrations.rst
    """
    return _annotation_repository.train_action_narrations()


def training_labels() -> pd.DataFrame:
    """
    Returns:
        Dataframe with the columns

        .. include:: meta/train_action_labels.rst
    """
    return _annotation_repository.train_action_labels().copy()


def training_object_labels() -> pd.DataFrame:
    """
    Returns:
        Dataframe with the columns

        .. include:: meta/train_object_labels.rst
    """
    return _annotation_repository.train_object_labels().copy()


def test_timestamps(split: str) -> pd.DataFrame:
    """
    Args:
        split: 'seen', 'unseen', or 'all' (loads both with a 'split'

    Returns:
        Dataframe with the columns

        .. include:: meta/test_timestamps.rst
    """
    if split == "all":
        return pd.concat(
            [
                _annotation_repository.test_seen_timestamps(),
                _annotation_repository.test_unseen_timestamps(),
            ]
        )

    if split == "seen":
        return _annotation_repository.test_seen_timestamps().copy()
    elif split == "unseen":
        return _annotation_repository.test_unseen_timestamps()
    raise ValueError("Unknown split '{}' expected 'seen' or 'unseen'".format(split))


def video_descriptions() -> pd.DataFrame:
    """
    Returns:
        High level description of the task trying to be accomplished in a video

        .. include:: meta/video_descriptions.rst
    """
    return _annotation_repository.video_descriptions().copy()


def video_info() -> pd.DataFrame:
    """
    Returns:
        Technical information stating the resolution, duration and FPS of each video.

        .. include:: meta/video_info.rst
    """
    return _annotation_repository.video_info().copy()
