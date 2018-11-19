import logging
import numpy as np
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


def numpy_to_builtin(obj):
    """
    Convert possibly numpy values into python builtin values. Traverses structures applying the type
    transformations

    Args:
        obj: an arbitrary python object, if it contains numpy values it will be converted into a
        corresponding python value.

    Examples:
        >>> type(numpy_to_builtin(np.int64(2)))
        <class 'int'>
        >>> type(numpy_to_builtin(np.int32(2)))
        <class 'int'>
        >>> type(numpy_to_builtin(np.int16(2)))
        <class 'int'>
        >>> type(numpy_to_builtin(np.int8(2)))
        <class 'int'>
        >>> type(numpy_to_builtin(np.int(2)))
        <class 'int'>
        >>> type(numpy_to_builtin(np.float(2.2)))
        <class 'float'>
        >>> l = [np.int8(2), [np.int64(3)]]
        >>> l_converted = numpy_to_builtin(l)
        >>> len(l) == len(l_converted)
        True
        >>> type(l_converted[0])
        <class 'int'>
        >>> type(l_converted[1][0])
        <class 'int'>
        >>> d = {'a': np.int8(2), 'b': { 'c': np.int32(2) }}
        >>> d_converted = numpy_to_builtin(d)
        >>> type(d_converted['a'])
        <class 'int'>
        >>> type(d_converted['b']['c'])
        <class 'int'>
        >>> type(numpy_to_builtin(np.array([1, 2], dtype=np.int32))[0])
        <class 'int'>
        >>> type(numpy_to_builtin(np.array([1.2, 2.2], dtype=np.float32))[0])
        <class 'float'>
    """
    if isinstance(obj, np.generic) and np.isscalar(obj):
        return np.asscalar(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        for key in obj.keys():
            value = obj[key]
            obj[key] = numpy_to_builtin(value)
        return obj
    if isinstance(obj, (list, tuple)):
        return [numpy_to_builtin(o) for o in obj]
    else:
        return obj
