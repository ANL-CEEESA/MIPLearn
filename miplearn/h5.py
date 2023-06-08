#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from types import TracebackType
from typing import Optional, Any, Union, List, Type, Literal

import h5py
import numpy as np
from scipy.sparse import coo_matrix

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


class H5File:
    def __init__(
        self,
        filename: str,
        mode: str = "r+",
    ) -> None:
        self.file = h5py.File(filename, mode, libver="latest")

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

    def put_scalar(self, key: str, value: Any) -> None:
        if value is None:
            return
        self._assert_is_scalar(value)
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(key, data=value)

    def put_array(self, key: str, value: Optional[np.ndarray]) -> None:
        if value is None:
            return
        self._assert_is_array(value)
        if value.dtype.kind == "f":
            value = value.astype("float32")
        if key in self.file:
            del self.file[key]
        return self.file.create_dataset(key, data=value, compression="gzip")

    def get_array(self, key: str) -> Optional[np.ndarray]:
        if key not in self.file:
            return None
        return self.file[key][:]

    def put_sparse(self, key: str, value: coo_matrix) -> None:
        if value is None:
            return
        self._assert_is_sparse(value)
        self.put_array(f"{key}_row", value.row)
        self.put_array(f"{key}_col", value.col)
        self.put_array(f"{key}_data", value.data)

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

    def close(self):
        self.file.close()

    def __enter__(self) -> "H5File":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        self.file.close()
        return False

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
        assert isinstance(
            value, coo_matrix
        ), f"coo_matrix expected; found: {value.__class__}"
        self._assert_is_array(value.data)
