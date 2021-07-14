#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from tempfile import NamedTemporaryFile
from typing import Any

from miplearn.features.sample import MemorySample, Sample, Hdf5Sample


def test_memory_sample() -> None:
    _test_sample(MemorySample())


def test_hdf5_sample() -> None:
    file = NamedTemporaryFile()
    _test_sample(Hdf5Sample(file.name))


def _test_sample(sample: Sample) -> None:
    # Scalar
    _assert_roundtrip_scalar(sample, "A")
    _assert_roundtrip_scalar(sample, True)
    _assert_roundtrip_scalar(sample, 1)
    _assert_roundtrip_scalar(sample, 1.0)

    # Vector
    _assert_roundtrip_vector(sample, ["A", "BB", "CCC", "こんにちは"])
    _assert_roundtrip_vector(sample, [True, True, False])
    _assert_roundtrip_vector(sample, [1, 2, 3])
    _assert_roundtrip_vector(sample, [1.0, 2.0, 3.0])

    # List[Optional[List[Primitive]]]
    # _assert_roundtrip(
    #     sample,
    #     [
    #         [1],
    #         None,
    #         [2, 2],
    #         [3, 3, 3],
    #     ],
    # )


def _assert_roundtrip_scalar(sample: Sample, expected: Any) -> None:
    sample.put_scalar("key", expected)
    actual = sample.get_scalar("key")
    assert actual == expected
    assert actual is not None
    _assert_same_type(actual, expected)


def _assert_roundtrip_vector(sample: Sample, expected: Any) -> None:
    sample.put_vector("key", expected)
    actual = sample.get_vector("key")
    assert actual == expected
    assert actual is not None
    _assert_same_type(actual[0], expected[0])


def _assert_same_type(actual: Any, expected: Any) -> None:
    assert isinstance(actual, expected.__class__), (
        f"Expected class {expected.__class__}, "
        f"found class {actual.__class__} instead"
    )
