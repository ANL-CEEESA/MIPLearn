#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import gzip
import logging
import pickle
from abc import ABC, abstractmethod
from typing import List, Union, cast, IO

import numpy as np
from tqdm.auto import tqdm

from miplearn.instance import Instance

logger = logging.getLogger(__name__)


class InstanceIterator:
    def __init__(
        self,
        instances: Union[List[str], List[Instance]],
    ) -> None:
        self.instances = instances
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self) -> Instance:
        if self.current >= len(self.instances):
            raise StopIteration
        result = self.instances[self.current]
        self.current += 1
        if isinstance(result, str):
            logger.debug("Read: %s" % result)
            try:
                if result.endswith(".gz"):
                    with gzip.GzipFile(result, "rb") as gzfile:
                        result = pickle.load(cast(IO[bytes], gzfile))
                else:
                    with open(result, "rb") as file:
                        result = pickle.load(cast(IO[bytes], file))
            except pickle.UnpicklingError:
                raise Exception(f"Invalid instance file: {result}")
        assert isinstance(result, Instance)
        return result


class Extractor(ABC):
    @abstractmethod
    def extract(self, instances):
        pass

    @staticmethod
    def split_variables(instance):
        result = {}
        lp_solution = instance.training_data[0]["LP solution"]
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
                        instance.training_data[0]["LP value"],
                    ]
                )
                for instance in InstanceIterator(instances)
            ]
        )
