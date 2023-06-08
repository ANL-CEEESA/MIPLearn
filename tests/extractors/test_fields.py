#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List

import pytest

from miplearn.extractors.fields import H5FieldsExtractor
from miplearn.h5 import H5File


def test_fields_instance(multiknapsack_h5: List[str]) -> None:
    ext = H5FieldsExtractor(
        instance_fields=[
            "lp_obj_value",
            "lp_var_values",
            "static_var_obj_coeffs",
        ],
        var_fields=["lp_var_values"],
    )
    with H5File(multiknapsack_h5[0], "r") as h5:
        x = ext.get_instance_features(h5)
        assert x.shape == (201,)

        x = ext.get_var_features(h5)
        assert x.shape == (100, 1)


def test_fields_instance_none(multiknapsack_h5: List[str]) -> None:
    ext = H5FieldsExtractor(instance_fields=None)
    with H5File(multiknapsack_h5[0], "r") as h5:
        with pytest.raises(Exception):
            ext.get_instance_features(h5)
