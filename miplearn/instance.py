#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import gzip
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Hashable, IO, cast

from miplearn.types import VarIndex
from miplearn.features import TrainingSample, Features

logger = logging.getLogger(__name__)


def write_pickle_gz(obj: Any, filename: str) -> None:
    logger.info(f"Writing: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.GzipFile(filename, "wb") as file:
        pickle.dump(obj, cast(IO[bytes], file))


def read_pickle_gz(filename: str) -> Any:
    logger.info(f"Reading: {filename}")
    with gzip.GzipFile(filename, "rb") as file:
        return pickle.load(cast(IO[bytes], file))


def write_pickle_gz_multiple(objs: List[Any], dirname: str) -> None:
    for (i, obj) in enumerate(objs):
        write_pickle_gz(obj, f"{dirname}/{i:05d}.pkl.gz")


# noinspection PyMethodMayBeStatic
class Instance(ABC):
    """
    Abstract class holding all the data necessary to generate a concrete model of the
    problem.

    In the knapsack problem, for example, this class could hold the number of items,
    their weights and costs, as well as the size of the knapsack. Objects
    implementing this class are able to convert themselves into a concrete
    optimization model, which can be optimized by a solver, or into arrays of
    features, which can be provided as inputs to machine learning models.
    """

    def __init__(self) -> None:
        self.training_data: List[TrainingSample] = []
        self.features: Features = Features()

    @abstractmethod
    def to_model(self) -> Any:
        """
        Returns the optimization model corresponding to this instance.
        """
        pass

    def get_instance_features(self) -> List[float]:
        """
        Returns a 1-dimensional array of (numerical) features describing the
        entire instance.

        The array is used by LearningSolver to determine how similar two instances
        are. It may also be used to predict, in combination with variable-specific
        features, the values of binary decision variables in the problem.

        There is not necessarily a one-to-one correspondence between models and
        instance features: the features may encode only part of the data necessary to
        generate the complete model. Features may also be statistics computed from
        the original data. For example, in the knapsack problem, an implementation
        may decide to provide as instance features only the average weights, average
        prices, number of items and the size of the knapsack.

        The returned array MUST have the same length for all relevant instances of
        the problem. If two instances map into arrays of different lengths,
        they cannot be solved by the same LearningSolver object.

        By default, returns [0].
        """
        return [0]

    def get_variable_features(self, var_name: str, index: VarIndex) -> List[float]:
        """
        Returns a 1-dimensional array of (numerical) features describing a particular
        decision variable.

        In combination with instance features, variable features are used by
        LearningSolver to predict, among other things, the optimal value of each
        decision variable before the optimization takes place. In the knapsack
        problem, for example, an implementation could provide as variable features
        the weight and the price of a specific item.

        Like instance features, the arrays returned by this method MUST have the same
        length for all variables within the same category, for all relevant instances
        of the problem.

        By default, returns [0].
        """
        return [0]

    def get_variable_category(
        self,
        var_name: str,
        index: VarIndex,
    ) -> Optional[Hashable]:
        """
        Returns the category for each decision variable.

        If two variables have the same category, LearningSolver will use the same
        internal ML model to predict the values of both variables. If the returned
        category is None, ML models will ignore the variable.

        By default, returns "default".
        """
        return "default"

    def get_constraint_features(self, cid: str) -> Optional[List[float]]:
        return [0.0]

    def get_constraint_category(self, cid: str) -> Optional[Hashable]:
        return cid

    def has_static_lazy_constraints(self) -> bool:
        return False

    def has_dynamic_lazy_constraints(self):
        return False

    def is_constraint_lazy(self, cid: str) -> bool:
        return False

    def find_violated_lazy_constraints(self, model):
        """
        Returns lazy constraint violations found for the current solution.

        After solving a model, LearningSolver will ask the instance to identify which
        lazy constraints are violated by the current solution. For each identified
        violation, LearningSolver will then call the build_lazy_constraint, add the
        generated Pyomo constraint to the model, then resolve the problem. The
        process repeats until no further lazy constraint violations are found.

        Each "violation" is simply a string, a tuple or any other hashable type which
        allows the instance to identify unambiguously which lazy constraint should be
        generated. In the Traveling Salesman Problem, for example, a subtour
        violation could be a frozen set containing the cities in the subtour.

        For a concrete example, see TravelingSalesmanInstance.
        """
        return []

    def build_lazy_constraint(self, model, violation):
        """
        Returns a Pyomo constraint which fixes a given violation.

        This method is typically called immediately after
        find_violated_lazy_constraints. The violation object provided to this method
        is exactly the same object returned earlier by
        find_violated_lazy_constraints. After some training, LearningSolver may
        decide to proactively build some lazy constraints at the beginning of the
        optimization process, before a solution is even available. In this case,
        build_lazy_constraints will be called without a corresponding call to
        find_violated_lazy_constraints.

        The implementation should not directly add the constraint to the model. The
        constraint will be added by LearningSolver after the method returns.

        For a concrete example, see TravelingSalesmanInstance.
        """
        pass

    def has_user_cuts(self) -> bool:
        return False

    def find_violated_user_cuts(self, model: Any) -> List[Hashable]:
        return []

    def build_user_cut(self, model: Any, violation: Hashable) -> Any:
        return None

    def flush(self) -> None:
        """
        Save any pending changes made to the instance to the underlying data store.
        """
        pass


def lazy_load(func):
    def inner(self, *args):
        if self.instance is None:
            self.instance = self._load()
            self.features = self.instance.features
            self.training_data = self.instance.training_data
        return func(self, *args)

    return inner


class PickleGzInstance(Instance):
    """
    An instance backed by a gzipped pickle file.

    The instance is only loaded to memory after an operation is called (for example,
    `to_model`).

    Parameters
    ----------
    filename: str
        Path of the gzipped pickle file that should be loaded.
    """

    # noinspection PyMissingConstructor
    def __init__(self, filename: str) -> None:
        assert os.path.exists(filename), f"File not found: {filename}"
        self.instance: Optional[Instance] = None
        self.filename: str = filename

    @lazy_load
    def to_model(self) -> Any:
        assert self.instance is not None
        return self.instance.to_model()

    @lazy_load
    def get_instance_features(self) -> List[float]:
        assert self.instance is not None
        return self.instance.get_instance_features()

    @lazy_load
    def get_variable_features(self, var_name: str, index: VarIndex) -> List[float]:
        assert self.instance is not None
        return self.instance.get_variable_features(var_name, index)

    @lazy_load
    def get_variable_category(
        self,
        var_name: str,
        index: VarIndex,
    ) -> Optional[Hashable]:
        assert self.instance is not None
        return self.instance.get_variable_category(var_name, index)

    @lazy_load
    def get_constraint_features(self, cid: str) -> Optional[List[float]]:
        assert self.instance is not None
        return self.instance.get_constraint_features(cid)

    @lazy_load
    def get_constraint_category(self, cid: str) -> Optional[Hashable]:
        assert self.instance is not None
        return self.instance.get_constraint_category(cid)

    @lazy_load
    def has_static_lazy_constraints(self) -> bool:
        assert self.instance is not None
        return self.instance.has_static_lazy_constraints()

    @lazy_load
    def has_dynamic_lazy_constraints(self):
        assert self.instance is not None
        return self.instance.has_dynamic_lazy_constraints()

    @lazy_load
    def is_constraint_lazy(self, cid: str) -> bool:
        assert self.instance is not None
        return self.instance.is_constraint_lazy(cid)

    @lazy_load
    def find_violated_lazy_constraints(self, model):
        assert self.instance is not None
        return self.instance.find_violated_lazy_constraints(model)

    @lazy_load
    def build_lazy_constraint(self, model, violation):
        assert self.instance is not None
        return self.instance.build_lazy_constraint(model, violation)

    @lazy_load
    def find_violated_user_cuts(self, model):
        assert self.instance is not None
        return self.instance.find_violated_user_cuts(model)

    @lazy_load
    def build_user_cut(self, model, violation):
        assert self.instance is not None
        return self.instance.build_user_cut(model, violation)

    def _load(self) -> Instance:
        obj = read_pickle_gz(self.filename)
        assert isinstance(obj, Instance)
        return obj

    def flush(self) -> None:
        write_pickle_gz(self.instance, self.filename)
