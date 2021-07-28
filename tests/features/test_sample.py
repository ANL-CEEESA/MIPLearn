#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np

from miplearn.features.sample import MemorySample, Sample, Hdf5Sample, _pad, _crop
from miplearn.solvers.tests import assert_equals


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
    _assert_roundtrip_vector(sample, ["A", "BB", "CCC", "こんにちは", None])
    _assert_roundtrip_vector(sample, [True, True, False])
    _assert_roundtrip_vector(sample, [1, 2, 3])
    _assert_roundtrip_vector(sample, [1.0, 2.0, 3.0])
    _assert_roundtrip_vector(sample, np.array([1.0, 2.0, 3.0]), check_type=False)

    # VectorList
    _assert_roundtrip_vector_list(sample, [["A"], ["BB", "CCC"], None])
    _assert_roundtrip_vector_list(sample, [[True], [False, False], None])
    _assert_roundtrip_vector_list(sample, [[1], None, [2, 2], [3, 3, 3]])
    _assert_roundtrip_vector_list(sample, [[1.0], None, [2.0, 2.0], [3.0, 3.0, 3.0]])
    _assert_roundtrip_vector_list(sample, [None, None])

    # Bytes
    _assert_roundtrip_bytes(sample, b"\x00\x01\x02\x03\x04\x05")

    # Querying unknown keys should return None
    assert sample.get_scalar("unknown-key") is None
    assert sample.get_vector("unknown-key") is None
    assert sample.get_vector_list("unknown-key") is None
    assert sample.get_bytes("unknown-key") is None

    # Putting None should not modify HDF5 file
    sample.put_scalar("key", None)
    sample.put_vector("key", None)


def _assert_roundtrip_bytes(sample: Sample, expected: Any) -> None:
    sample.put_bytes("key", expected)
    actual = sample.get_bytes("key")
    assert actual == expected
    assert actual is not None
    _assert_same_type(actual, expected)


def _assert_roundtrip_scalar(sample: Sample, expected: Any) -> None:
    sample.put_scalar("key", expected)
    actual = sample.get_scalar("key")
    assert actual == expected
    assert actual is not None
    _assert_same_type(actual, expected)


def _assert_roundtrip_vector(
    sample: Sample, expected: Any, check_type: bool = True
) -> None:
    sample.put_vector("key", expected)
    actual = sample.get_vector("key")
    assert_equals(actual, expected)
    assert actual is not None
    if check_type:
        _assert_same_type(actual[0], expected[0])


def _assert_roundtrip_vector_list(sample: Sample, expected: Any) -> None:
    sample.put_vector_list("key", expected)
    actual = sample.get_vector_list("key")
    assert actual == expected
    assert actual is not None
    if actual[0] is not None:
        _assert_same_type(actual[0][0], expected[0][0])


def _assert_same_type(actual: Any, expected: Any) -> None:
    assert isinstance(
        actual, expected.__class__
    ), f"Expected {expected.__class__}, found {actual.__class__} instead"


def test_pad_int() -> None:
    _assert_roundtrip_pad(
        original=[[1], [2, 2, 2], [], [3, 3], [4, 4, 4, 4], None],
        expected_padded=[
            [1, 0, 0, 0],
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [3, 3, 0, 0],
            [4, 4, 4, 4],
            [0, 0, 0, 0],
        ],
        expected_lens=[1, 3, 0, 2, 4, -1],
        dtype=int,
    )


def test_pad_float() -> None:
    _assert_roundtrip_pad(
        original=[[1.0], [2.0, 2.0, 2.0], [3.0, 3.0], [4.0, 4.0, 4.0, 4.0], None],
        expected_padded=[
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        expected_lens=[1, 3, 2, 4, -1],
        dtype=float,
    )


def test_pad_str() -> None:
    _assert_roundtrip_pad(
        original=[["A"], ["B", "B", "B"], ["C", "C"]],
        expected_padded=[["A", "", ""], ["B", "B", "B"], ["C", "C", ""]],
        expected_lens=[1, 3, 2],
        dtype=str,
    )


def _assert_roundtrip_pad(
    original: Any,
    expected_padded: Any,
    expected_lens: Any,
    dtype: Any,
) -> None:
    actual_padded, actual_lens = _pad(original)
    assert actual_padded == expected_padded
    assert actual_lens == expected_lens
    for v in actual_padded:
        for vi in v:  # type: ignore
            assert isinstance(vi, dtype)
    cropped = _crop(actual_padded, actual_lens)
    assert cropped == original
    for v in cropped:
        if v is None:
            continue
        for vi in v:  # type: ignore
            assert isinstance(vi, dtype)
