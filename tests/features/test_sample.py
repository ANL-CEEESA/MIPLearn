#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from miplearn.features.sample import MemorySample, Sample, Hdf5Sample


def test_memory_sample() -> None:
    _test_sample(MemorySample())


def test_hdf5_sample() -> None:
    file = NamedTemporaryFile()
    _test_sample(Hdf5Sample(file.name))


def _test_sample(sample: Sample) -> None:
    _assert_roundtrip_scalar(sample, "A")
    _assert_roundtrip_scalar(sample, True)
    _assert_roundtrip_scalar(sample, 1)
    _assert_roundtrip_scalar(sample, 1.0)
    _assert_roundtrip_array(sample, np.array([True, False], dtype="bool"))
    _assert_roundtrip_array(sample, np.array([1, 2, 3], dtype="int16"))
    _assert_roundtrip_array(sample, np.array([1, 2, 3], dtype="int32"))
    _assert_roundtrip_array(sample, np.array([1, 2, 3], dtype="int64"))
    _assert_roundtrip_array(sample, np.array([1.0, 2.0, 3.0], dtype="float16"))
    _assert_roundtrip_array(sample, np.array([1.0, 2.0, 3.0], dtype="float32"))
    _assert_roundtrip_array(sample, np.array([1.0, 2.0, 3.0], dtype="float64"))
    _assert_roundtrip_array(sample, np.array(["A", "BB", "CCC"], dtype="S"))
    assert sample.get_scalar("unknown-key") is None
    assert sample.get_array("unknown-key") is None


def _assert_roundtrip_array(sample: Sample, expected: Any) -> None:
    sample.put_array("key", expected)
    actual = sample.get_array("key")
    assert actual is not None
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == expected.dtype
    assert (actual == expected).all()


def _assert_roundtrip_scalar(sample: Sample, expected: Any) -> None:
    sample.put_scalar("key", expected)
    actual = sample.get_scalar("key")
    assert actual == expected
    assert actual is not None
    assert isinstance(
        actual, expected.__class__
    ), f"Expected {expected.__class__}, found {actual.__class__} instead"
