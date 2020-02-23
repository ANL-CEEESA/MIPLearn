#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

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
    
    @staticmethod
    def merge(partial_results, vertical=False):
        results = {}
        all_categories = set()
        for pr in partial_results:
            all_categories |= pr.keys()
        for category in all_categories:
            results[category] = []
            for pr in partial_results:
                if category in pr.keys():
                    results[category] += [pr[category]]
            if vertical:
                results[category] = np.vstack(results[category])
            else:
                results[category] = np.hstack(results[category])
        return results


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
    def __init__(self, relaxation=False):
        self.relaxation = relaxation
        
    def extract(self, instances, models=None):
        result = {}
        if models is None:
            models = [instance.to_model() for instance in instances]
        for (index, instance) in enumerate(instances):
            model = models[index]
            var_split = self.split_variables(instance, model)
            for (category, var_index_pairs) in var_split.items():
                if category not in result.keys():
                    result[category] = []
                for (var, index) in var_index_pairs:
                    if self.relaxation:
                        v = instance.lp_solution[str(var)][index]
                    else:
                        v = instance.solution[str(var)][index]
                    if v is None:
                        result[category] += [[0, 0]]
                    else:
                        result[category] += [[1 - v, v]]
        for category in result.keys():
            result[category] = np.vstack(result[category])            
        return result
    
    
class CombinedExtractor(Extractor):
    def __init__(self, extractors):
        self.extractors = extractors
    
    def extract(self, instances, models):
        return self.merge([ex.extract(instances, models)
                           for ex in self.extractors])
