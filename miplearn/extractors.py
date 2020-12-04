#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import pickle
import gzip

import numpy as np

from tqdm.auto import tqdm
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class InstanceIterator:
    def __init__(self, instances):
        self.instances = instances
        self.current = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= len(self.instances):
            raise StopIteration
        result = self.instances[self.current]
        self.current += 1
        if isinstance(result, str):
            logger.info("Read: %s" % result)
            if result.endswith(".gz"):
                with gzip.GzipFile(result, "rb") as file:
                    result = pickle.load(file)
            else:
                with open(result, "rb") as file:
                    result = pickle.load(file)
        return result


class Extractor(ABC):
    @abstractmethod
    def extract(self, instances,):
        pass
    
    @staticmethod
    def split_variables(instance):
        assert hasattr(instance, "lp_solution")
        result = {}
        for var_name in instance.lp_solution:
            for index in instance.lp_solution[var_name]:
                category = instance.get_variable_category(var_name, index)
                if category is None:
                    continue
                if category not in result:
                    result[category] = []
                result[category] += [(var_name, index)]
        return result

    
class VariableFeaturesExtractor(Extractor):
    def extract(self, instances):
        result = {}
        for instance in tqdm(InstanceIterator(instances),
                             desc="Extract (vars)",
                             disable=len(instances) < 5):
            instance_features = instance.get_instance_features()
            var_split = self.split_variables(instance)
            for (category, var_index_pairs) in var_split.items():
                if category not in result:
                    result[category] = []
                for (var_name, index) in var_index_pairs:
                    result[category] += [
                        instance_features.tolist() + \
                        instance.get_variable_features(var_name, index).tolist() + \
                        [instance.lp_solution[var_name][index]]
                    ]
        for category in result:
            result[category] = np.array(result[category])
        return result


class SolutionExtractor(Extractor):
    def __init__(self, relaxation=False):
        self.relaxation = relaxation
        
    def extract(self, instances):
        result = {}
        for instance in tqdm(InstanceIterator(instances),
                             desc="Extract (solution)",
                             disable=len(instances) < 5):
            var_split = self.split_variables(instance)
            for (category, var_index_pairs) in var_split.items():
                if category not in result:
                    result[category] = []
                for (var_name, index) in var_index_pairs:
                    if self.relaxation:
                        v = instance.lp_solution[var_name][index]
                    else:
                        v = instance.solution[var_name][index]
                    if v is None:
                        result[category] += [[0, 0]]
                    else:
                        result[category] += [[1 - v, v]]
        for category in result:
            result[category] = np.array(result[category])            
        return result
    
    
class InstanceFeaturesExtractor(Extractor):
    def extract(self, instances):
        return np.vstack([
            np.hstack([
                instance.get_instance_features(),
                instance.lp_value,
            ])
            for instance in InstanceIterator(instances)
        ])
    
    
class ObjectiveValueExtractor(Extractor):
    def __init__(self, kind="lp"):
        assert kind in ["lower bound", "upper bound", "lp"]
        self.kind = kind
        
    def extract(self, instances):
        if self.kind == "lower bound":
            return np.array([[instance.lower_bound]
                             for instance in InstanceIterator(instances)])
        if self.kind == "upper bound":
            return np.array([[instance.upper_bound]
                             for instance in InstanceIterator(instances)])
        if self.kind == "lp":
            return np.array([[instance.lp_value]
                             for instance in InstanceIterator(instances)])
