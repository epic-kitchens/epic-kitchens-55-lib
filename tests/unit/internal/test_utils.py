import urllib.request
from pathlib import Path
from unittest.mock import Mock, MagicMock

from epic_kitchens.internal.utils import before, maybe_download


URL_ROOT = "https://raw.githubusercontent.com/epic-kitchens/annotations/master/"


def test_before_decorator():
    fn = lambda x: "blah"
    before_fn = Mock()
    decorator = before(before_fn)
    wrapped_fn = decorator(fn)

    assert wrapped_fn("x") == "blah"
    before_fn.assert_called_once_with()


def test_maybe_download_when_file_doesnt_exist(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    urlretrieve = MagicMock()
    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)

    url = URL_ROOT + "EPIC_verb_classes.csv"
    maybe_download(url, "file")

    urlretrieve.assert_called_once_with(url, "file")


def test_maybe_download_when_file_does_exist(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    urlretrieve = MagicMock()
    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)

    url = URL_ROOT + "EPIC_verb_classes.csv"
    maybe_download(url, "file")

    urlretrieve.assert_not_called()
