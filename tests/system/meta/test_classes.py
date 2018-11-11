import pytest

from epic_kitchens.meta import classes
from epic_kitchens.meta.classes import ActionClass


def test_reading_verb_classes(tmp_datadir):
    verb_classes = classes.verb_classes()
    assert verb_classes["class_key"].loc[0] == "take"


def test_reading_noun_classes(tmp_datadir):
    noun_classes = classes.noun_classes()
    assert noun_classes["class_key"].loc[0] == "Nothing"


def test_reading_many_shot_verbs(tmp_datadir):
    many_shot_verbs = classes.many_shot_verbs()
    assert 0 in many_shot_verbs


def test_reading_many_shot_nouns(tmp_datadir):
    many_shot_nouns = classes.many_shot_nouns()
    assert 1 in many_shot_nouns


def test_reading_many_shot_actions(tmp_datadir):
    many_shot_actions = classes.many_shot_actions()
    assert ActionClass(0, 1) in many_shot_actions
