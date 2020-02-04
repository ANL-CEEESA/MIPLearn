# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import numpy as np
from abc import ABC, abstractmethod
from pyomo.core import Var


class Extractor(ABC):
    @abstractmethod
    def extract(self, instances, models):
        pass
    
    @staticmethod
    def split_variables(instance, model):
        result = {}
        for var in model.component_objects(Var):
            for index in var:
                category = instance.get_variable_category(var, index)
                if category is None:
                    continue
                if category not in result.keys():
                    result[category] = []
                result[category] += [(var, index)]
        return result


class UserFeaturesExtractor(Extractor):
    def extract(self,
                instances,
                models=None,
               ):
        result = {}
        if models is None:
            models = [instance.to_model() for instance in instances]
        for (index, instance) in enumerate(instances):
            model = models[index]
            instance_features = instance.get_instance_features()
            var_split = self.split_variables(instance, model)
            for (category, var_index_pairs) in var_split.items():
                if category not in result.keys():
                    result[category] = []
                for (var, index) in var_index_pairs:
                    result[category] += [np.hstack([
                        instance_features,
                        instance.get_variable_features(var, index),
                    ])]
        for category in result.keys():
            result[category] = np.vstack(result[category])
        return result


class SolutionExtractor(Extractor):
    def extract(self, instances, models):
        result = {}
        for (index, instance) in enumerate(instances):
            model = models[index]
            var_split = self.split_variables(instance, model)
            for (category, var_index_pairs) in var_split.items():
                if category not in result.keys():
                    result[category] = []
                for (var, index) in var_index_pairs:
                    result[category] += [[
                        1 - var[index].value,
                        var[index].value,
                    ]]
        for category in result.keys():
            result[category] = np.vstack(result[category])            
        return result