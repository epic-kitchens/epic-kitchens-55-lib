from functools import reduce

import pytest

from epic_kitchens import meta


@pytest.fixture(scope="module")
def training_labels(tmp_datadir):
    return meta.training_labels()


@pytest.fixture(scope="module")
def training_object_labels(tmp_datadir):
    return meta.training_object_labels()


def test_training_action_labels_has_more_than_20000_items(training_labels):
    assert len(training_labels) >= 20000


def test_training_action_labels_has_uid_index(training_labels):
    assert training_labels.index.name == "uid"


@pytest.mark.parametrize(
    "col",
    [
        "video_id",
        "narration",
        "start_timestamp",
        "stop_timestamp",
        "participant_id",
        "verb",
        "noun",
        "verb_class",
        "noun_class",
        "all_nouns",
        "all_noun_classes",
    ],
)
def test_training_action_labels_has_column(col, training_labels):
    assert col in training_labels.columns


@pytest.mark.parametrize(
    "col",
    ["noun", "noun_class", "frame", "participant_id", "video_id", "bounding_boxes"],
)
def test_training_object_labels_has_column(col, training_object_labels):
    assert col in training_object_labels.columns


def test_training_object_labels_has_over_350_000_rows(training_object_labels):
    assert len(training_object_labels) >= 350000


def test_training_object_labels_has_list_of_bounding_boxes_per_row(
    training_object_labels
):
    assert (
        training_object_labels["bounding_boxes"]
        .apply(lambda bbs: isinstance(bbs, list))
        .all()
    )


def test_training_object_labels_has_4_tuple_bounding_boxes(training_object_labels):
    def all_fours(all_fours_so_far: bool, bounding_boxes):
        return all_fours_so_far and len(bounding_boxes) == 4

    assert (
        training_object_labels["bounding_boxes"]
        .apply(lambda bbs: reduce(all_fours, bbs, True))
        .all()
    )
