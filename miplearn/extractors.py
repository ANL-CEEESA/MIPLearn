#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Extractor(ABC):
    @abstractmethod
    def extract(self, instances):
        pass

    @staticmethod
    def split_variables(instance):
        result = {}
        lp_solution = instance.training_data[0].lp_solution
        for var_name in lp_solution:
            for index in lp_solution[var_name]:
                category = instance.get_variable_category(var_name, index)
                if category is None:
                    continue
                if category not in result:
                    result[category] = []
                result[category] += [(var_name, index)]
        return result


class InstanceFeaturesExtractor(Extractor):
    def extract(self, instances):
        return np.vstack(
            [
                np.hstack(
                    [
                        instance.get_instance_features(),
                        instance.training_data[0].lp_value,
                    ]
                )
                for instance in instances
            ]
        )
