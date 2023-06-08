#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Optional, List

import numpy as np

from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.h5 import H5File


class H5FieldsExtractor(FeaturesExtractor):
    def __init__(
        self,
        instance_fields: Optional[List[str]] = None,
        var_fields: Optional[List[str]] = None,
        constr_fields: Optional[List[str]] = None,
    ):
        self.instance_fields = instance_fields
        self.var_fields = var_fields
        self.constr_fields = constr_fields

    def get_instance_features(self, h5: H5File) -> np.ndarray:
        if self.instance_fields is None:
            raise Exception("No instance fields provided")
        x = []
        for field in self.instance_fields:
            try:
                data = h5.get_array(field)
            except ValueError:
                data = h5.get_scalar(field)
            assert data is not None
            x.append(data)
        x = np.hstack(x)
        assert len(x.shape) == 1
        return x

    def get_var_features(self, h5: H5File) -> np.ndarray:
        var_types = h5.get_array("static_var_types")
        assert var_types is not None
        n_vars = len(var_types)
        if self.var_fields is None:
            raise Exception("No var fields provided")
        return self._extract(h5, self.var_fields, n_vars)

    def get_constr_features(self, h5: H5File) -> np.ndarray:
        constr_sense = h5.get_array("static_constr_sense")
        assert constr_sense is not None
        n_constr = len(constr_sense)
        if self.constr_fields is None:
            raise Exception("No constr fields provided")
        return self._extract(h5, self.constr_fields, n_constr)

    def _extract(self, h5, fields, n_expected):
        x = []
        for field in fields:
            try:
                data = h5.get_array(field)
            except ValueError:
                v = h5.get_scalar(field)
                data = np.repeat(v, n_expected)
            assert data is not None
            assert len(data.shape) == 1
            assert data.shape[0] == n_expected
            x.append(data)
        features = np.vstack(x).T
        assert len(features.shape) == 2
        assert features.shape[0] == n_expected
        return features
