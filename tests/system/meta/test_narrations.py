import pytest
import epic_kitchens.meta


@pytest.fixture(scope="module")
def training_narrations(tmp_datadir):
    return epic_kitchens.meta.training_narrations()


def test_training_narrations_has_at_least_20_000_entries(training_narrations):
    assert len(training_narrations) >= 20000


@pytest.mark.parametrize(
    "col",
    ["participant_id", "video_id", "start_timestamp", "stop_timestamp", "narration"],
)
def test_training_narrations_has_columns(col, training_narrations):
    assert col in training_narrations.columns
