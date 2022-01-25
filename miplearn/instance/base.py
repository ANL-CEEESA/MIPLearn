#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING, Dict

import numpy as np

from miplearn.features.sample import Sample, MemorySample
from miplearn.types import ConstraintName

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import InternalSolver


# noinspection PyMethodMayBeStatic
class Instance(ABC):
    """
    Abstract class holding all the data necessary to generate a concrete model of the
    proble.

    In the knapsack problem, for example, this class could hold the number of items,
    their weights and costs, as well as the size of the knapsack. Objects
    implementing this class are able to convert themselves into a concrete
    optimization model, which can be optimized by a solver, or into arrays of
    features, which can be provided as inputs to machine learning models.
    """

    def __init__(self) -> None:
        self._samples: List[Sample] = []

    @abstractmethod
    def to_model(self) -> Any:
        """
        Returns the optimization model corresponding to this instance.
        """
        pass

    def get_instance_features(self) -> np.ndarray:
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

        By default, returns [0.0].
        """
        return np.zeros(1)

    def get_variable_features(self, names: np.ndarray) -> np.ndarray:
        """
        Returns dictionary mapping the name of each variable to a (1-dimensional) list
        of numerical features describing a particular decision variable.

        In combination with instance features, variable features are used by
        LearningSolver to predict, among other things, the optimal value of each
        decision variable before the optimization takes place. In the knapsack
        problem, for example, an implementation could provide as variable features
        the weight and the price of a specific item.

        Like instance features, the arrays returned by this method MUST have the same
        length for all variables within the same category, for all relevant instances
        of the problem.

        If features are not provided for a given variable, MIPLearn will use a
        default set of features.

        By default, returns [[0.0], ..., [0.0]].
        """
        return np.zeros((len(names), 1))

    def get_variable_categories(self, names: np.ndarray) -> np.ndarray:
        """
        Returns a dictionary mapping the name of each variable to its category.

        If two variables have the same category, LearningSolver will use the same
        internal ML model to predict the values of both variables. If a variable is not
        listed in the dictionary, ML models will ignore the variable.

        By default, returns `names`.
        """
        return names

    def get_constraint_features(self, names: np.ndarray) -> np.ndarray:
        return np.zeros((len(names), 1))

    def get_constraint_categories(self, names: np.ndarray) -> np.ndarray:
        return names

    def has_dynamic_lazy_constraints(self) -> bool:
        return False

    def are_constraints_lazy(self, names: np.ndarray) -> np.ndarray:
        return np.zeros(len(names), dtype=bool)

    def find_violated_lazy_constraints(
        self,
        solver: "InternalSolver",
        model: Any,
    ) -> Dict[ConstraintName, Any]:
        """
        Returns lazy constraint violations found for the current solution.

        After solving a model, LearningSolver will ask the instance to identify which
        lazy constraints are violated by the current solution. For each identified
        violation, LearningSolver will then call the enforce_lazy_constraint and
        resolve the problem. The process repeats until no further lazy constraint
        violations are found.

        Violations should be returned in a dictionary mapping the name of the violation
        to some user-specified data that allows the instance to unambiguously generate
        the lazy constraints at a later time. In the Traveling Salesman Problem, for
        example, this function could return a dictionary identifying violated subtour
        inequalities. More concretely, it could return:
            {
                "s1": [1, 2, 3],
                "s2": [4, 5, 6, 7],
            }
        where "s1" and "s2" are the names of the subtours, and [1,2,3] and [4,5,6,7]
        are the cities in each subtour. The names of the violations should be kept
        stable across instances. In our example, "s1" should always correspond to
        [1,2,3] across all instances. The user-provided data should be picklable.

        The current solution can be queried with `solver.get_solution()`. If the solver
        is configured to use lazy callbacks, this solution may be non-integer.

        For a concrete example, see TravelingSalesmanInstance.
        """
        return {}

    def enforce_lazy_constraint(
        self,
        solver: "InternalSolver",
        model: Any,
        violation_data: Any,
    ) -> None:
        """
        Adds constraints to the model to ensure that the given violation is fixed.

        This method is typically called immediately after
        `find_violated_lazy_constraints`. The argument `violation_data` is the
        user-provided data, previously returned by `find_violated_lazy_constraints`.
        In the Traveling Salesman Problem, for example, it could be a list of cities
        in the subtour.

        After some training, LearningSolver may decide to proactively build some lazy
        constraints at the beginning of the optimization process, before a solution
        is even available. In this case, `enforce_lazy_constraints` will be called
        without a corresponding call to `find_violated_lazy_constraints`.

        For a concrete example, see TravelingSalesmanInstance.
        """
        pass

    def has_user_cuts(self) -> bool:
        return False

    def find_violated_user_cuts(self, model: Any) -> Dict[ConstraintName, Any]:
        return {}

    def enforce_user_cut(
        self,
        solver: "InternalSolver",
        model: Any,
        violation_data: Any,
    ) -> Any:
        return None

    def load(self) -> None:
        pass

    def free(self) -> None:
        pass

    def flush(self) -> None:
        """
        Save any pending changes made to the instance to the underlying data store.
        """
        pass

    def get_samples(self) -> List[Sample]:
        return self._samples

    def create_sample(self) -> Sample:
        sample = MemorySample()
        self._samples.append(sample)
        return sample
