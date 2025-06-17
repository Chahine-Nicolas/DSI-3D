import cv2
import numpy as np
from os.path import splitext
from typing import Any, Generator, Iterator, Optional, Tuple, Union
from pathlib import Path
import os
import os.path as osp
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import tempfile
import pickle
import os
import json


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    _allow_symlink = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def allow_symlink(self):
        return self._allow_symlink

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    _allow_symlink = True

    def __init__(self, **kwargs):
        pass

    def get(self, filepath: Union[str, Path], update_cache: bool = False) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8',
                 update_cache: bool = False) -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def put(self, obj: bytes, filepath: Union[str, Path], update_cache: bool = False) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        mkdir_or_exist(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            f.write(obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8',
                 update_cache: bool = False) -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        mkdir_or_exist(os.path.dirname(filepath))
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return os.path.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        return os.path.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        return os.path.isfile(filepath)

    @contextmanager
    def get_local_path(
            self,
            filepath: Union[str, Path],
            update_cache: bool = False) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        yield filepath

    def load_pickle(self, filepath, update_cache: bool = False):
        return pickle.load(open(filepath, 'rb'))

    def dump_pickle(self, data, filepath, update_cache: bool = False):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def save_npy(self, data, filepath, update_cache: bool = False):
        np.save(filepath, data)

    def load_npy(self, filepath, update_cache: bool = False):
        return np.load(filepath)
    
    def load_to_numpy(self, filepath, dtype, update_cache: bool = False):
        return np.fromfile(filepath, dtype=dtype)

    def load_img(self, filepath, update_cache: bool = False):
        return cv2.imread(filepath, cv2.IMREAD_COLOR)

    def load_json(self, filepath, update_cache: bool = False):
        return json.load(open(filepath, 'r'))
   
    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:

        if list_dir and suffix is not None:
            raise TypeError('`suffix` should be None when `list_dir` is True')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith('.') and entry.is_file():
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(entry.path, list_dir,
                                                     list_file, suffix,
                                                     recursive)

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)
