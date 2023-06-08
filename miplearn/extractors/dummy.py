#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import numpy as np

from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.h5 import H5File


class DummyExtractor(FeaturesExtractor):
    def get_instance_features(self, h5: H5File) -> np.ndarray:
        return np.zeros(1)

    def get_var_features(self, h5: H5File) -> np.ndarray:
        var_types = h5.get_array("static_var_types")
        assert var_types is not None
        n_vars = len(var_types)
        return np.zeros((n_vars, 1))

    def get_constr_features(self, h5: H5File) -> np.ndarray:
        constr_sense = h5.get_array("static_constr_sense")
        assert constr_sense is not None
        n_constr = len(constr_sense)
        return np.zeros((n_constr, 1))
