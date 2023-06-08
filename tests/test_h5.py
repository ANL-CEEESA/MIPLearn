#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix

from miplearn.h5 import H5File


def test_h5() -> None:
    file = NamedTemporaryFile()
    h5 = H5File(file.name)
    _assert_roundtrip_scalar(h5, "A")
    _assert_roundtrip_scalar(h5, True)
    _assert_roundtrip_scalar(h5, 1)
    _assert_roundtrip_scalar(h5, 1.0)
    assert h5.get_scalar("unknown-key") is None

    _assert_roundtrip_array(h5, np.array([True, False]))
    _assert_roundtrip_array(h5, np.array([1, 2, 3]))
    _assert_roundtrip_array(h5, np.array([1.0, 2.0, 3.0]))
    _assert_roundtrip_array(h5, np.array(["A", "BB", "CCC"], dtype="S"))
    assert h5.get_array("unknown-key") is None

    _assert_roundtrip_sparse(
        h5,
        coo_matrix(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 3.0],
                [0.0, 0.0, 4.0],
            ],
        ),
    )
    assert h5.get_sparse("unknown-key") is None


def _assert_roundtrip_array(h5: H5File, original: np.ndarray) -> None:
    h5.put_array("key", original)
    recovered = h5.get_array("key")
    assert recovered is not None
    assert isinstance(recovered, np.ndarray)
    assert (recovered == original).all()


def _assert_roundtrip_scalar(h5: H5File, original: Any) -> None:
    h5.put_scalar("key", original)
    recovered = h5.get_scalar("key")
    assert recovered == original
    assert recovered is not None
    assert isinstance(
        recovered, original.__class__
    ), f"Expected {original.__class__}, found {recovered.__class__} instead"


def _assert_roundtrip_sparse(h5: H5File, original: coo_matrix) -> None:
    h5.put_sparse("key", original)
    recovered = h5.get_sparse("key")
    assert recovered is not None
    assert isinstance(recovered, coo_matrix)
    assert (original != recovered).sum() == 0
