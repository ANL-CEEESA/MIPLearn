#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Optional, Any, Union, List, Tuple, cast, Set
from scipy.sparse import coo_matrix

import h5py
import numpy as np
from h5py import Dataset
from overrides import overrides

Bytes = Union[bytes, bytearray]
Scalar = Union[None, bool, str, int, float]
Vector = Union[
    None,
    List[bool],
    List[str],
    List[int],
    List[float],
    List[Optional[str]],
    np.ndarray,
]
VectorList = Union[
    List[List[bool]],
    List[List[str]],
    List[List[int]],
    List[List[float]],
    List[Optional[List[bool]]],
    List[Optional[List[str]]],
    List[Optional[List[int]]],
    List[Optional[List[float]]],
]


class Sample(ABC):
    """Abstract dictionary-like class that stores training data."""

    @abstractmethod
    def get_scalar(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def put_scalar(self, key: str, value: Scalar) -> None:
        pass

    @abstractmethod
    def put_array(self, key: str, value: Optional[np.ndarray]) -> None:
        pass

    @abstractmethod
    def get_array(self, key: str) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def put_sparse(self, key: str, value: coo_matrix) -> None:
        pass

    @abstractmethod
    def get_sparse(self, key: str) -> Optional[coo_matrix]:
        pass

    def _assert_is_scalar(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, bool, int, float, bytes, np.bytes_)):
            return
        assert False, f"scalar expected; found instead: {value} ({value.__class__})"

    def _assert_is_array(self, value: np.ndarray) -> None:
        assert isinstance(
            value, np.ndarray
        ), f"np.ndarray expected; found instead: {value.__class__}"
        assert value.dtype.kind in "biufS", f"Unsupported dtype: {value.dtype}"

    def _assert_is_sparse(self, value: Any) -> None:
        assert isinstance(value, coo_matrix)
        self._assert_is_array(value.data)


class MemorySample(Sample):
    """Dictionary-like class that stores training data in-memory."""

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if data is None:
            data = {}
        self._data: Dict[str, Any] = data

    @overrides
    def get_scalar(self, key: str) -> Optional[Any]:
        return self._get(key)

    @overrides
    def put_scalar(self, key: str, value: Scalar) -> None:
        if value is None:
            return
        self._assert_is_scalar(value)
        self._put(key, value)

    def _get(self, key: str) -> Optional[Any]:
        if key in self._data:
            return self._data[key]
        else:
            return None

    def _put(self, key: str, value: Any) -> None:
        self._data[key] = value

    @overrides
    def put_array(self, key: str, value: Optional[np.ndarray]) -> None:
        if value is None:
            return
        self._assert_is_array(value)
        self._put(key, value)

    @overrides
    def get_array(self, key: str) -> Optional[np.ndarray]:
        return cast(Optional[np.ndarray], self._get(key))

    @overrides
    def put_sparse(self, key: str, value: coo_matrix) -> None:
        if value is None:
            return
        self._assert_is_sparse(value)
        self._put(key, value)

    @overrides
    def get_sparse(self, key: str) -> Optional[coo_matrix]:
        return cast(Optional[coo_matrix], self._get(key))


class Hdf5Sample(Sample):
    """
    Dictionary-like class that stores training data in an HDF5 file.

    Unlike MemorySample, this class only loads to memory the parts of the data set that
    are actually accessed, and therefore it is more scalable.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "r+",
    ) -> None:
        self.file = h5py.File(filename, mode, libver="latest")

    @overrides
    def get_scalar(self, key: str) -> Optional[Any]:
        if key not in self.file:
            return None
        ds = self.file[key]
        assert (
            len(ds.shape) == 0
        ), f"0-dimensional array expected; found shape {ds.shape}"
        if h5py.check_string_dtype(ds.dtype):
            return ds.asstr()[()]
        else:
            return ds[()].tolist()

    @overrides
    def put_scalar(self, key: str, value: Any) -> None:
        if value is None:
            return
        self._assert_is_scalar(value)
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(key, data=value)

    @overrides
    def put_array(self, key: str, value: Optional[np.ndarray]) -> None:
        if value is None:
            return
        self._assert_is_array(value)
        if len(value.shape) > 1 and value.dtype.kind == "f":
            value = value.astype("float16")
        if key in self.file:
            del self.file[key]
        return self.file.create_dataset(key, data=value, compression="gzip")

    @overrides
    def get_array(self, key: str) -> Optional[np.ndarray]:
        if key not in self.file:
            return None
        return self.file[key][:]

    @overrides
    def put_sparse(self, key: str, value: coo_matrix) -> None:
        if value is None:
            return
        self._assert_is_sparse(value)
        self.put_array(f"{key}_row", value.row)
        self.put_array(f"{key}_col", value.col)
        self.put_array(f"{key}_data", value.data)

    @overrides
    def get_sparse(self, key: str) -> Optional[coo_matrix]:
        row = self.get_array(f"{key}_row")
        if row is None:
            return None
        col = self.get_array(f"{key}_col")
        data = self.get_array(f"{key}_data")
        assert col is not None
        assert data is not None
        return coo_matrix((data, (row, col)))

    def get_bytes(self, key: str) -> Optional[Bytes]:
        if key not in self.file:
            return None
        ds = self.file[key]
        assert (
            len(ds.shape) == 1
        ), f"1-dimensional array expected; found shape {ds.shape}"
        return ds[()].tobytes()

    def put_bytes(self, key: str, value: Bytes) -> None:
        assert isinstance(
            value, (bytes, bytearray)
        ), f"bytes expected; found: {value.__class__}"  # type: ignore
        self.put_array(key, np.frombuffer(value, dtype="uint8"))
