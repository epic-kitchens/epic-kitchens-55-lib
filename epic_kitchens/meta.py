import pandas as pd
from collections import namedtuple

from epic_kitchens.internal.loading import AnnotationRepository

from pathlib import Path
from typing import Union, Set
from logging import getLogger

_LOG = getLogger(__name__)

_annotation_repository = AnnotationRepository()


ActionClass = namedtuple("ActionClass", ["verb_class", "noun_class"])
Action = namedtuple("Action", ["verb", "noun"])


def set_datadir(dir_: Union[str, Path]):
    """
    Set download directory

    Parameters
    ----------
    dir_: Path to directory in which to store all downloaded metadata files

    """
    _annotation_repository.local_dir = Path(dir_)
    _LOG.info("Setting data directory to {}".format(dir_))


def get_datadir() -> Path:
    """

    Returns
    -------
    datadir
        Directory under which any downloaded files are stored, defaults to current working directory

    """
    return _annotation_repository.local_dir


def verb_to_class(verb: str) -> int:
    """
    Parameters
    ----------
    verb: A noun from a narration

    Returns
    -------
    class: The corresponding numeric class of the verb if it exists

    Raises
    ------
    IndexError: If the verb doesn't belong to any of the verb classes

    """
    return _annotation_repository.inverse_verb_lookup()[verb]


def noun_to_class(noun: str) -> int:
    """

    Parameters
    ----------
    noun: A noun from a narration

    Returns
    -------
    class: The corresponding numeric class of the noun if it exists

    Raises
    ------
    IndexError: If the noun doesn't belong to any of the noun classes

    """
    return _annotation_repository.inverse_noun_lookup()[noun]


def class_to_verb(cls: int) -> str:
    """

    Parameters
    ----------
    cls: numeric verb class

    Returns
    -------
    canonical verb representing the class


    Raises
    ------
    IndexError: if ``cls`` is an invalid verb class
    """
    return _annotation_repository.verb_classes()["class_key"].loc[cls]


def class_to_noun(cls: int) -> str:
    """

    Parameters
    ----------
    cls: numeric noun class

    Returns
    -------
    canonical noun representing the class

    Raises
    ------
    IndexError: if ``cls`` is an invalid noun class
    """
    return _annotation_repository.noun_classes()["class_key"].loc[cls]


def noun_classes() -> pd.DataFrame:
    """
    Get dataframe containing the mapping between numeric noun classes, the canonical noun of that class
    and nouns clustered into the class.

    Returns
    -------
    ``pd.DataFrame`` with the columns:
        ``index``: int, the numeric noun class
        ``class_key``: str, canonical noun representing the class as a whole
        ``nouns``: [str], the list of nouns that are clustered into this class.
    """
    return _annotation_repository.noun_classes().copy()


def verb_classes() -> pd.DataFrame:
    """
    Get dataframe containing the mapping between numeric verb classes, the canonical verb of that class
    and verbs clustered into the class.

    Returns
    -------
    ``pd.DataFrame`` with the columns:
        ``index``: int, the numeric verb class
        ``class_key``: str, canonical verb representing the class as a whole
        ``verbs``: [str], the list of verbs that are clustered into this class.
    """
    return _annotation_repository.verb_classes().copy()


def many_shot_verbs() -> Set[int]:
    """

    Returns
    -------
    many_shot_verbs
        The set of verb classes that are many shot (appear more than 100 times in training).

    """
    return set(_annotation_repository.many_shot_verbs())


def many_shot_nouns() -> Set[int]:
    """

    Returns
    -------
    many_shot_nouns
        The set of noun classes that are many shot (appear more than 100 times in training).

    """
    return set(_annotation_repository.many_shot_nouns())


def many_shot_actions() -> Set[ActionClass]:
    """

    Returns
    -------
    many_shot_actions
        The set of actions classes that are many shot (verb_class appears more than 100 times
        in training, noun_class appears more than 100 times in training, and the action appears
        at least once in training).

    """
    return set(_annotation_repository.many_shot_actions())


def is_many_shot_action(action_class: ActionClass) -> bool:
    return action_class in _annotation_repository.many_shot_actions()


def is_many_shot_verb(verb_class: int) -> bool:
    return verb_class in _annotation_repository.many_shot_verbs()


def is_many_shot_noun(noun_class: int) -> bool:
    return noun_class in _annotation_repository.many_shot_nouns()


def training_narrations() -> pd.DataFrame:
    return _annotation_repository.train_action_narrations()


def training_labels() -> pd.DataFrame:
    """

    Returns
    -------
    training_labels : pd.DataFrame
        Metadata describing the training action annotations with the following rows


    """
    return _annotation_repository.train_action_labels().copy()


def training_object_labels() -> pd.DataFrame:
    """

    Returns
    -------
    training_object_labels : pd.DataFrame
        Metadata describing the training object annotations with the following rows


    """
    return _annotation_repository.train_object_labels().copy()


def test_timestamps(split: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    split: 'seen', 'unseen', or 'all' (loads both with a 'split'

    Returns
    -------
    timestamps
        Timestamps of test action segments with the following rows

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

    Returns
    -------
    video_descriptions
        High level description of the task trying to be accomplished in a video

    """
    return _annotation_repository.video_descriptions().copy()


def video_info() -> pd.DataFrame:
    """

    Returns
    -------
    video_info
        Technical information stating the resolution, duration and FPS of each video.

    """
    return _annotation_repository.video_info().copy()
