import pytest

from epic_kitchens import meta


@pytest.fixture(scope="module")
def test_seen_timestamps(tmp_datadir):
    return meta.test_timestamps("seen")


@pytest.fixture(scope="module")
def test_unseen_timestamps(tmp_datadir):
    return meta.test_timestamps("unseen")


@pytest.fixture(scope="module")
def test_timestamps(test_seen_timestamps, test_unseen_timestamps):
    return meta.test_timestamps("all")


def test_seen_split_has_more_than_8000_items(test_seen_timestamps):
    assert len(test_seen_timestamps) >= 8000


def test_unseen_split_has_more_than_1000_items(test_unseen_timestamps):
    assert len(test_unseen_timestamps) >= 1000


def test_seen_split_has_uid_index(test_seen_timestamps):
    assert test_seen_timestamps.index.name == "uid"


def test_unseen_split_has_uid_index(test_unseen_timestamps):
    assert test_unseen_timestamps.index.name == "uid"


def test_all_timestamps_is_as_long_as_sum_of_both_timestamp_splits(
    test_timestamps, test_unseen_timestamps, test_seen_timestamps
):
    assert len(test_timestamps) == len(test_unseen_timestamps) + len(
        test_seen_timestamps
    )


timestamp_columns = ["video_id", "start_timestamp", "stop_timestamp", "participant_id"]


@pytest.mark.parametrize("col", timestamp_columns)
def test_seen_timestamp_has_column(col, test_seen_timestamps):
    assert col in test_seen_timestamps.columns


@pytest.mark.parametrize("col", timestamp_columns)
def test_unseen_timestamp_has_column(col, test_unseen_timestamps):
    assert col in test_unseen_timestamps.columns
