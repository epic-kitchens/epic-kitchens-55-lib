import os
import pathlib

import pytest
from pathlib import Path
from unittest.mock import MagicMock
import urllib.request

from epic_kitchens.internal.loading import Loaders, HttpFolder, AnnotationRepository

base_url = "https://github.com/epic-kitchens/annotations/raw/v1.5.0/"
file_name = "EPIC_video_info.csv"
loaders = {file_name: MagicMock()}


@pytest.fixture(scope="function")
def http_folder(tmpdir_factory):
    local_dir = Path(str(tmpdir_factory.mktemp("v1.5.0")))
    folder = HttpFolder(base_url, local_dir, loaders)
    yield folder
    for loader in loaders.values():
        loader.reset_mock()


@pytest.fixture(scope="function")
def dummy_path_mkdir(monkeypatch):
    with monkeypatch.context() as ctx:
        ctx.setattr(Path, "mkdir", lambda self, **kwargs: None)
        yield


@pytest.fixture(scope="function")
def dummy_urlretrieve(monkeypatch):
    with monkeypatch.context() as ctx:
        ctx.setattr(urllib.request, "urlretrieve", MagicMock())
        yield


@pytest.mark.usefixtures("dummy_path_mkdir", "dummy_urlretrieve")
class TestHttpFolder:
    def test_load_file_downloads_file_if_it_doesnt_exist(
        self, monkeypatch, http_folder
    ):
        with monkeypatch.context() as ctx:
            urlretrieve = MagicMock()
            ctx.setattr(urllib.request, "urlretrieve", urlretrieve)
            http_folder.load_file(file_name)

        urlretrieve.assert_called_once_with(
            base_url + file_name, str(http_folder.local_dir / file_name)
        )

    def test_load_file_uses_loader_for_file(self, monkeypatch, http_folder):
        http_folder.load_file(file_name)

        loaders[file_name].assert_called_once_with(http_folder.local_dir / file_name)

    def test_load_file_caches_file_load(self, monkeypatch, http_folder):
        with monkeypatch.context() as ctx:
            http_folder.load_file(file_name)
            http_folder.load_file(file_name)

        loaders[file_name].assert_called_once_with(http_folder.local_dir / file_name)

    def test_load_file_creates_parent_dir_if_it_doesnt_exist(
        self, monkeypatch, http_folder
    ):
        mkdir_mock = MagicMock()
        with monkeypatch.context() as ctx:
            ctx.setattr(Path, "mkdir", mkdir_mock)
            http_folder.load_file(file_name)
        mkdir_mock.assert_called_once_with(exist_ok=True, parents=True)


class TestLoaders:
    def test_loader_defaults_to_filetype_loader_for_unknown_file(self):
        filetype_loaders = {"pkl": MagicMock()}
        loaders = Loaders(dict(), filetype_loaders)
        assert loaders["unknown_file.pkl"] == filetype_loaders["pkl"]

    def test_loader_uses_file_specific_loader_for_known_file(self):
        filetype_loaders = {"pkl": MagicMock()}
        filename = "EPIC_train_action_labels.pkl"
        file_loaders = {filename: MagicMock()}
        loaders = Loaders(file_loaders, filetype_loaders)

        assert loaders[filename] == file_loaders[filename]


class TestAnnotationRepository:
    def test_defaults_to_user_cache_dir_for_storage_location(self):
        repo = AnnotationRepository()
        assert repo.http_folder.local_dir == Path.home() / "epic_kitchens" / "v1.5.0"

    def test_uses_version_string_in_storage_location(self):
        version = "v0.0.1"
        repo = AnnotationRepository(version=version)
        assert version in str(repo.http_folder.local_dir)

    def test_uses_version_string_in_url(self):
        version = "v0.0.1"
        repo = AnnotationRepository(version=version)
        assert repo.http_folder.base_url.endswith(version + "/")

    def test_uses_XDG_CACHE_HOME_if_set_for_storage_location(self, monkeypatch):
        cache_dir = "/home/will/.mycache"
        with monkeypatch.context() as ctx:
            ctx.setattr(os.environ, "get", lambda name, default: cache_dir)
            repo = AnnotationRepository()
        assert cache_dir in str(repo.http_folder.local_dir)
