#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Hashable, TYPE_CHECKING

from overrides import EnforceOverrides

from miplearn.features import TrainingSample, Features, Sample
from miplearn.types import VariableName, Category

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import InternalSolver


# noinspection PyMethodMayBeStatic
class Instance(ABC, EnforceOverrides):
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
        self.training_data: List[TrainingSample] = []
        self.features: Features = Features()
        self.samples: List[Sample] = []

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

    def get_variable_features(self, var_name: VariableName) -> List[float]:
        """
        Returns a (1-dimensional) list of numerical features describing a particular
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

    def get_variable_category(self, var_name: VariableName) -> Optional[Category]:
        """
        Returns the category for each decision variable.

        If two variables have the same category, LearningSolver will use the same
        internal ML model to predict the values of both variables. If the returned
        category is None, ML models will ignore the variable.

        A category can be any hashable type, such as strings, numbers or tuples.
        By default, returns "default".
        """
        return "default"

    def get_constraint_features(self, cid: str) -> List[float]:
        return [0.0]

    def get_constraint_category(self, cid: str) -> Optional[Hashable]:
        return cid

    def has_static_lazy_constraints(self) -> bool:
        return False

    def has_dynamic_lazy_constraints(self) -> bool:
        return False

    def is_constraint_lazy(self, cid: str) -> bool:
        return False

    def find_violated_lazy_constraints(
        self,
        solver: "InternalSolver",
        model: Any,
    ) -> List[Hashable]:
        """
        Returns lazy constraint violations found for the current solution.

        After solving a model, LearningSolver will ask the instance to identify which
        lazy constraints are violated by the current solution. For each identified
        violation, LearningSolver will then call the enforce_lazy_constraint and
        resolve the problem. The process repeats until no further lazy constraint
        violations are found.

        Each "violation" is simply a string, a tuple or any other hashable type which
        allows the instance to identify unambiguously which lazy constraint should be
        generated. In the Traveling Salesman Problem, for example, a subtour
        violation could be a frozen set containing the cities in the subtour.

        The current solution can be queried with `solver.get_solution()`. If the solver
        is configured to use lazy callbacks, this solution may be non-integer.

        For a concrete example, see TravelingSalesmanInstance.
        """
        return []

    def enforce_lazy_constraint(
        self,
        solver: "InternalSolver",
        model: Any,
        violation: Hashable,
    ) -> None:
        """
        Adds constraints to the model to ensure that the given violation is fixed.

        This method is typically called immediately after
        find_violated_lazy_constraints. The violation object provided to this method
        is exactly the same object returned earlier by
        find_violated_lazy_constraints. After some training, LearningSolver may
        decide to proactively build some lazy constraints at the beginning of the
        optimization process, before a solution is even available. In this case,
        enforce_lazy_constraints will be called without a corresponding call to
        find_violated_lazy_constraints.

        Note that this method can be called either before the optimization starts or
        from within a callback. To ensure that constraints are added correctly in
        either case, it is recommended to use `solver.add_constraint`, instead of
        modifying the `model` object directly.

        For a concrete example, see TravelingSalesmanInstance.
        """
        pass

    def has_user_cuts(self) -> bool:
        return False

    def find_violated_user_cuts(self, model: Any) -> List[Hashable]:
        return []

    def enforce_user_cut(
        self,
        solver: "InternalSolver",
        model: Any,
        violation: Hashable,
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
