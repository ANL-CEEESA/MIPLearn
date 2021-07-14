#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from tempfile import NamedTemporaryFile
from typing import Any

from miplearn.features.sample import MemorySample, Sample, Hdf5Sample


def _test_sample(sample: Sample) -> None:
    _assert_roundtrip(sample, "A")
    _assert_roundtrip(sample, True)
    _assert_roundtrip(sample, 1)
    _assert_roundtrip(sample, 1.0)
    _assert_roundtrip(sample, ["A", "BB", "CCC", "こんにちは"])
    _assert_roundtrip(sample, [True, True, False])
    _assert_roundtrip(sample, [1, 2, 3])
    _assert_roundtrip(sample, [1.0, 2.0, 3.0])


def _assert_roundtrip(sample: Sample, expected: Any) -> None:
    sample.put("key", expected)
    actual = sample.get("key")
    assert actual == expected
    assert actual is not None
    if isinstance(actual, list):
        assert isinstance(actual[0], expected[0].__class__), (
            f"Expected class {expected[0].__class__}, "
            f"found {actual[0].__class__} instead"
        )
    else:
        assert isinstance(actual, expected.__class__), (
            f"Expected class {expected.__class__}, "
            f"found class {actual.__class__} instead"
        )


def test_memory_sample() -> None:
    _test_sample(MemorySample())


def test_hdf5_sample() -> None:
    file = NamedTemporaryFile()
    _test_sample(Hdf5Sample(file.name))
