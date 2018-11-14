from datetime import datetime

import epic_kitchens.meta
import pytest


@pytest.fixture(scope="module")
def descriptions(tmp_datadir):
    return epic_kitchens.meta.video_descriptions()


def test_descriptions_has_at_least_432_entries(descriptions):
    assert len(descriptions) >= 432


def test_descriptions_has_video_id_index(descriptions):
    assert descriptions.index.name == "video_id"


@pytest.mark.parametrize("col", ["datetime", "description"])
def test_description_has_columns(col, descriptions):
    assert col in descriptions.columns


def test_descriptions_reads_date_and_time_to_datetime(descriptions):
    assert isinstance(descriptions["datetime"].iloc[0], datetime)
