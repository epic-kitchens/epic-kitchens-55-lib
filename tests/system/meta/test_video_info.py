import pytest

import epic_kitchens.meta


@pytest.fixture(scope="module")
def video_info(tmp_datadir):
    return epic_kitchens.meta.video_info()


def test_video_info_has_video_id_index(video_info):
    assert video_info.index.name == "video_id"


def test_video_info_has_at_least_432_entries(video_info):
    assert len(video_info) >= 432


@pytest.mark.parametrize("col", ["resolution", "duration", "fps"])
def test_video_info_has_columns(col, video_info):
    assert col in video_info.columns
