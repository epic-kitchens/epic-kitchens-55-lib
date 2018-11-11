import logging
import urllib.request
from pathlib import Path

import functools
from typing import Union


def before(before_fn):
    """
    Build a decorator function that will apply ``before_fn`` before calling the decorated function.
    """

    def decorator(fn):
        def wrapped_fn(*args, **kwargs):
            before_fn()
            return fn(*args, **kwargs)

        return functools.update_wrapper(wrapped_fn, fn)

    return decorator


def maybe_download(url: str, filepath: Union[Path, str]):
    filepath = Path(filepath)
    if not filepath.exists():
        _LOG.info("Downloading {} to {}".format(url, filepath))
        urllib.request.urlretrieve(url, str(filepath))


_LOG = logging.getLogger(__name__)
