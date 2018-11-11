import pytest

from epic_kitchens.meta import set_datadir


@pytest.fixture(scope="session")
def tmp_datadir(tmpdir_factory):
    set_datadir(str(tmpdir_factory.getbasetemp()))
