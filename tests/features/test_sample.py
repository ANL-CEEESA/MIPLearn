#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix

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
    assert sample.get_scalar("unknown-key") is None

    _assert_roundtrip_array(sample, np.array([True, False], dtype="bool"))
    _assert_roundtrip_array(sample, np.array([1, 2, 3], dtype="int16"))
    _assert_roundtrip_array(sample, np.array([1, 2, 3], dtype="int32"))
    _assert_roundtrip_array(sample, np.array([1, 2, 3], dtype="int64"))
    _assert_roundtrip_array(sample, np.array([1.0, 2.0, 3.0], dtype="float16"))
    _assert_roundtrip_array(sample, np.array([1.0, 2.0, 3.0], dtype="float32"))
    _assert_roundtrip_array(sample, np.array([1.0, 2.0, 3.0], dtype="float64"))
    _assert_roundtrip_array(sample, np.array(["A", "BB", "CCC"], dtype="S"))
    assert sample.get_array("unknown-key") is None

    _assert_roundtrip_sparse(
        sample,
        coo_matrix(
            [
                [1, 0, 0],
                [0, 2, 3],
                [0, 0, 4],
            ],
            dtype=float,
        ),
    )
    assert sample.get_sparse("unknown-key") is None


def _assert_roundtrip_array(sample: Sample, original: np.ndarray) -> None:
    sample.put_array("key", original)
    recovered = sample.get_array("key")
    assert recovered is not None
    assert isinstance(recovered, np.ndarray)
    assert recovered.dtype == original.dtype
    assert (recovered == original).all()


def _assert_roundtrip_scalar(sample: Sample, original: Any) -> None:
    sample.put_scalar("key", original)
    recovered = sample.get_scalar("key")
    assert recovered == original
    assert recovered is not None
    assert isinstance(
        recovered, original.__class__
    ), f"Expected {original.__class__}, found {recovered.__class__} instead"


def _assert_roundtrip_sparse(sample: Sample, original: coo_matrix) -> None:
    sample.put_sparse("key", original)
    recovered = sample.get_sparse("key")
    assert recovered is not None
    assert isinstance(recovered, coo_matrix)
    assert recovered.dtype == original.dtype
    assert (original != recovered).sum() == 0
