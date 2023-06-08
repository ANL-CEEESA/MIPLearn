#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from glob import glob
from os.path import dirname
from typing import List

import pytest

from miplearn.extractors.fields import H5FieldsExtractor
from miplearn.extractors.abstract import FeaturesExtractor


@pytest.fixture()
def multiknapsack_h5() -> List[str]:
    return sorted(glob(f"{dirname(__file__)}/fixtures/multiknapsack*.h5"))


@pytest.fixture()
def default_extractor() -> FeaturesExtractor:
    return H5FieldsExtractor(
        instance_fields=["static_var_obj_coeffs"],
        var_fields=["lp_var_features"],
    )
