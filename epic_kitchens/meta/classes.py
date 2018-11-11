import pandas as pd
from ast import literal_eval
from collections import namedtuple

from copy import copy
from typing import Dict, Set

from .._utils import before, maybe_download
from . import _urls, get_datadir


ActionClass = namedtuple("ActionClass", ["verb_class", "noun_class"])
Action = namedtuple("Action", ["verb", "noun"])

_verb_classes = None  # type: ignore
_inverse_verb_lookup = None  # type: ignore

_noun_classes = None  # type: ignore
_inverse_noun_lookup = None  # type: ignore

_many_shot_verbs = None  # type: ignore
_many_shot_nouns = None  # type: ignore
_many_shot_actions = None  # type: ignore


def _initialize_verbs() -> None:
    global _verb_classes, _inverse_verb_lookup
    file_path = get_datadir() / "EPIC_verb_classes.csv"
    if _verb_classes is None:
        maybe_download(_urls.verb_classes_url, file_path)
        _verb_classes = pd.read_csv(
            file_path, index_col="verb_id", converters={"verbs": literal_eval}
        )
    if _inverse_verb_lookup is None:
        _inverse_verb_lookup = _construct_word_to_class_dict(_verb_classes, "verbs")


def _initialize_nouns() -> None:
    global _noun_classes, _inverse_noun_lookup
    if _noun_classes is None:
        file_path = get_datadir() / "EPIC_noun_classes.csv"
        maybe_download(_urls.noun_classes_url, file_path)
        _noun_classes = pd.read_csv(
            file_path, index_col="noun_id", converters={"nouns": literal_eval}
        )
    if _inverse_noun_lookup is None:
        _inverse_noun_lookup = _construct_word_to_class_dict(_noun_classes, "nouns")


def _construct_word_to_class_dict(
    classes: pd.DataFrame, cluster_members_col: str
) -> Dict[str, int]:
    word_to_class = dict()
    for cls, row in classes.iterrows():
        for word in row[cluster_members_col]:
            word_to_class[word] = cls
    return word_to_class


def _initialize_many_shot_actions() -> None:
    global _many_shot_actions
    if _many_shot_actions is None:
        file_path = get_datadir() / "EPIC_many_shot_actions.csv"
        maybe_download(_urls.many_shot_actions_url, file_path)
        many_shot_actions_df = pd.read_csv(
            file_path,
            index_col="action_class",
            converters={"action_class": literal_eval},
        )
        _many_shot_actions = set(many_shot_actions_df.index.values)


def _initialize_many_shot_verbs() -> None:
    global _many_shot_verbs
    if _many_shot_verbs is None:
        file_path = get_datadir() / "EPIC_many_shot_verbs.csv"
        maybe_download(_urls.many_shot_verbs_url, file_path)
        many_shot_verbs_df = pd.read_csv(file_path, index_col="verb_class")
        _many_shot_verbs = set(many_shot_verbs_df.index.values)


def _initialize_many_shot_nouns() -> None:
    global _many_shot_nouns
    if _many_shot_nouns is None:
        file_path = get_datadir() / "EPIC_many_shot_nouns.csv"
        maybe_download(_urls.many_shot_nouns_url, file_path)
        many_shot_nouns_df = pd.read_csv(file_path, index_col="noun_class")
        _many_shot_nouns = set(many_shot_nouns_df.index.values)


requires_verbs = before(_initialize_verbs)
requires_nouns = before(_initialize_nouns)
requires_many_shot_actions = before(_initialize_many_shot_actions)
requires_many_shot_verbs = before(_initialize_many_shot_verbs)
requires_many_shot_nouns = before(_initialize_many_shot_nouns)


@requires_verbs
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
    assert _inverse_verb_lookup is not None
    return _inverse_verb_lookup[verb]


@requires_nouns
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
    assert _inverse_noun_lookup is not None
    return _inverse_noun_lookup[noun]


@requires_verbs
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
    assert _verb_classes is not None
    return _verb_classes["class_key"].loc[cls]


@requires_nouns
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
    assert _noun_classes is not None
    return _noun_classes["class_key"].loc[cls]


@requires_nouns
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
    assert _noun_classes is not None
    return _noun_classes.copy()


@requires_verbs
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
    assert _verb_classes is not None
    return _verb_classes.copy()


@requires_many_shot_verbs
def many_shot_verbs() -> Set[int]:
    """

    Returns
    -------

    """
    assert _many_shot_verbs is not None
    return set(_many_shot_verbs)


@requires_many_shot_nouns
def many_shot_nouns() -> Set[int]:
    """

    Returns
    -------

    """
    assert _many_shot_nouns is not None
    return set(_many_shot_nouns)


@requires_many_shot_actions
def many_shot_actions() -> Set[ActionClass]:
    """

    Returns
    -------

    """
    assert _many_shot_actions is not None
    return set(_many_shot_actions)


@requires_many_shot_actions
def is_many_shot_action(action_class: ActionClass) -> bool:
    assert _many_shot_actions is not None
    return action_class in _many_shot_actions


@requires_many_shot_verbs
def is_many_shot_verb(verb_class: int) -> bool:
    assert _many_shot_verbs is not None
    return verb_class in _many_shot_verbs


@requires_many_shot_nouns
def is_many_shot_noun(noun_class: int) -> bool:
    assert _many_shot_nouns is not None
    return noun_class in _many_shot_nouns
