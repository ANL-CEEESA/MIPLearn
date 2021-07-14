#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Union, List

import h5py
from overrides import overrides

Scalar = Union[None, bool, str, int, float]
Vector = Union[None, List[bool], List[str], List[int], List[float]]


class Sample(ABC):
    """Abstract dictionary-like class that stores training data."""

    @abstractmethod
    def get_scalar(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def put_scalar(self, key: str, value: Scalar) -> None:
        pass

    @abstractmethod
    def get_vector(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def put_vector(self, key: str, value: Vector) -> None:
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """
        Add a new key/value pair to the sample. If the key already exists,
        the previous value is silently replaced.

        Only the following data types are supported:
        - str, bool, int, float
        - List[str], List[bool], List[int], List[float]
        """
        pass

    def _assert_supported(self, value: Any) -> None:
        def _is_primitive(v: Any) -> bool:
            if isinstance(v, (str, bool, int, float)):
                return True
            if v is None:
                return True
            return False

        if _is_primitive(value):
            return
        if isinstance(value, list):
            if _is_primitive(value[0]):
                return
            if isinstance(value[0], list):
                if _is_primitive(value[0][0]):
                    return
        assert False, f"Value has unsupported type: {value}"

    def _assert_scalar(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, bool, int, float)):
            return
        assert False, f"Scalar expected; found instead: {value}"

    def _assert_vector(self, value: Any) -> None:
        assert isinstance(value, list), f"List expected; found instead: {value}"
        for v in value:
            self._assert_scalar(v)


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
        return self.get(key)

    @overrides
    def put_scalar(self, key: str, value: Scalar) -> None:
        self._assert_scalar(value)
        self.put(key, value)

    @overrides
    def get_vector(self, key: str) -> Optional[Any]:
        return self.get(key)

    @overrides
    def put_vector(self, key: str, value: Vector) -> None:
        if value is None:
            return
        self._assert_vector(value)
        self.put(key, value)

    @overrides
    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            return self._data[key]
        else:
            return None

    @overrides
    def put(self, key: str, value: Any) -> None:
        self._data[key] = value


class Hdf5Sample(Sample):
    """
    Dictionary-like class that stores training data in an HDF5 file.

    Unlike MemorySample, this class only loads to memory the parts of the data set that
    are actually accessed, and therefore it is more scalable.
    """

    def __init__(self, filename: str) -> None:
        self.file = h5py.File(filename, "r+")

    @overrides
    def get_scalar(self, key: str) -> Optional[Any]:
        ds = self.file[key]
        assert len(ds.shape) == 0
        if h5py.check_string_dtype(ds.dtype):
            return ds.asstr()[()]
        else:
            return ds[()].tolist()

    @overrides
    def get_vector(self, key: str) -> Optional[Any]:
        ds = self.file[key]
        assert len(ds.shape) == 1
        if h5py.check_string_dtype(ds.dtype):
            return ds.asstr()[:].tolist()
        else:
            return ds[:].tolist()

    @overrides
    def put_scalar(self, key: str, value: Any) -> None:
        self._assert_scalar(value)
        self.put(key, value)

    @overrides
    def put_vector(self, key: str, value: Vector) -> None:
        if value is None:
            return
        self._assert_vector(value)
        self.put(key, value)

    @overrides
    def get(self, key: str) -> Optional[Any]:
        ds = self.file[key]
        if h5py.check_string_dtype(ds.dtype):
            return ds.asstr()[:].tolist()
        else:
            return ds[:].tolist()

    @overrides
    def put(self, key: str, value: Any) -> None:
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(key, data=value)
