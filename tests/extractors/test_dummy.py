#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import List

from miplearn.extractors.dummy import DummyExtractor
from miplearn.h5 import H5File


def test_dummy(multiknapsack_h5: List[str]) -> None:
    ext = DummyExtractor()
    with H5File(multiknapsack_h5[0], "r") as h5:
        x = ext.get_instance_features(h5)
        assert x.shape == (1,)
        x = ext.get_var_features(h5)
        assert x.shape == (100, 1)
        x = ext.get_constr_features(h5)
        assert x.shape == (4, 1)
