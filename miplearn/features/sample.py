#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import h5py
import numpy as np
from overrides import overrides


class Sample(ABC):
    """Abstract dictionary-like class that stores training data."""

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
            return False

        if _is_primitive(value):
            return
        if isinstance(value, list):
            if _is_primitive(value[0]):
                return
        assert False, f"Value has unsupported type: {value}"


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
    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            return self._data[key]
        else:
            return None

    @overrides
    def put(self, key: str, value: Any) -> None:
        # self._assert_supported(value)
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
    def get(self, key: str) -> Optional[Any]:
        ds = self.file[key]
        if h5py.check_string_dtype(ds.dtype):
            if ds.shape == ():
                return ds.asstr()[()]
            else:
                return ds.asstr()[:].tolist()
        else:
            if ds.shape == ():
                return ds[()].tolist()
            else:
                return ds[:].tolist()

    @overrides
    def put(self, key: str, value: Any) -> None:
        self._assert_supported(value)
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(key, data=value)
