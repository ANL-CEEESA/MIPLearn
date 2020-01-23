# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import numpy as np
from pyomo.core import Var


class PerVariableTransformer:
    """
    Class that converts a miplearn.Instance into a matrix of features that is suitable
    for training machine learning models that make one decision per decision variable.
    """

    def __init__(self):
        pass

    def transform_instance(self, instance, var_index_pairs):
        instance_features = self._get_instance_features(instance)
        variable_features = self._get_variable_features(instance, var_index_pairs)
        return np.vstack([
            np.hstack([instance_features, vf])
            for vf in variable_features
        ])

    @staticmethod
    def _get_instance_features(instance):
        features = instance.get_instance_features()
        assert isinstance(features, np.ndarray)
        return features

    @staticmethod
    def _get_variable_features(instance, var_index_pairs):
        features = []
        expected_shape = None
        for (var, index) in var_index_pairs:
            vf = instance.get_variable_features(var, index)
            assert isinstance(vf, np.ndarray)
            if expected_shape is None:
                assert len(vf.shape) == 1
                expected_shape = vf.shape
            else:
                assert vf.shape == expected_shape
            features += [vf]
        return np.array(features)

    @staticmethod
    def transform_solution(var_index_pairs):
        y = []
        for (var, index) in var_index_pairs:
            y += [[1 - var[index].value, var[index].value]]
        return np.array(y)

    @staticmethod
    def split_variables(instance, model):
        result = {}
        for var in model.component_objects(Var):
            for index in var:
                category = instance.get_variable_category(var, index)
                if category not in result.keys():
                    result[category] = []
                result[category] += [(var, index)]
        return result
