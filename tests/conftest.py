#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import os
import shutil
import tempfile
from glob import glob
from os.path import dirname, basename, isfile
from tempfile import NamedTemporaryFile
from typing import List, Any

import pytest

from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.extractors.fields import H5FieldsExtractor


def _h5_fixture(pattern: str, request: Any) -> List[str]:
    """
    Create a temporary copy of the provided .h5 files, along with the companion
    .pkl.gz files, and return the path to the copy. Also register a finalizer,
    so that the temporary folder is removed after the tests.
    """
    filenames = glob(f"{dirname(__file__)}/fixtures/{pattern}")
    print(filenames)
    tmpdir = tempfile.mkdtemp()

    def cleanup() -> None:
        shutil.rmtree(tmpdir)

    request.addfinalizer(cleanup)

    print(tmpdir)
    for f in filenames:
        fbase, _ = os.path.splitext(f)
        for ext in [".h5", ".pkl.gz"]:
            dest = os.path.join(tmpdir, f"{basename(fbase)}{ext}")
            print(dest)
            shutil.copy(f"{fbase}{ext}", dest)
            assert isfile(dest)
    return sorted(glob(f"{tmpdir}/*.h5"))


@pytest.fixture()
def multiknapsack_h5(request: Any) -> List[str]:
    return _h5_fixture("multiknapsack*.h5", request)


@pytest.fixture()
def tsp_h5(request: Any) -> List[str]:
    return _h5_fixture("tsp*.h5", request)


@pytest.fixture()
def stab_h5(request: Any) -> List[str]:
    return _h5_fixture("stab*.h5", request)


@pytest.fixture()
def default_extractor() -> FeaturesExtractor:
    return H5FieldsExtractor(
        instance_fields=["static_var_obj_coeffs"],
        var_fields=["lp_var_features"],
    )
