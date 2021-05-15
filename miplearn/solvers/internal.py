#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from miplearn.features import Constraint, VariableFeatures, ConstraintFeatures
from miplearn.instance.base import Instance
from miplearn.types import (
    IterationCallback,
    LazyCallback,
    BranchPriorities,
    UserCutCallback,
    Solution,
)

logger = logging.getLogger(__name__)


@dataclass
class LPSolveStats:
    lp_log: Optional[str] = None
    lp_value: Optional[float] = None
    lp_wallclock_time: Optional[float] = None

    def to_list(self) -> List[float]:
        features: List[float] = []
        for attr in ["lp_value", "lp_wallclock_time"]:
            if getattr(self, attr) is not None:
                features.append(getattr(self, attr))
        return features


@dataclass
class MIPSolveStats:
    mip_lower_bound: Optional[float] = None
    mip_log: Optional[str] = None
    mip_nodes: Optional[int] = None
    mip_sense: Optional[str] = None
    mip_upper_bound: Optional[float] = None
    mip_wallclock_time: Optional[float] = None
    mip_warm_start_value: Optional[float] = None


class InternalSolver(ABC):
    """
    Abstract class representing the MIP solver used internally by LearningSolver.
    """

    @abstractmethod
    def add_constraints(self, cf: ConstraintFeatures) -> None:
        """Adds the given constraints to the model."""
        pass

    @abstractmethod
    def are_constraints_satisfied(
        self,
        cf: ConstraintFeatures,
        tol: float = 1e-5,
    ) -> Tuple[bool, ...]:
        """
        Checks whether the current solution satisfies the given constraints.
        """
        pass

    def are_callbacks_supported(self) -> bool:
        """
        Returns True if this solver supports native callbacks, such as lazy constraints
        callback or user cuts callback.
        """
        return False

    @abstractmethod
    def build_test_instance_infeasible(self) -> Instance:
        pass

    @abstractmethod
    def build_test_instance_knapsack(self) -> Instance:
        """
        Returns an instance corresponding to the following MIP, for testing purposes:

          maximize  505 x0 + 352 x1 + 458 x2 + 220 x3
          s.t.      eq_capacity: z = 23 x0 + 26 x1 + 20 x2 + 18 x3
                    x0, x1, x2, x3 binary
                    0 <= z <= 67 continuous
        """
        pass

    @abstractmethod
    def build_test_instance_redundancy(self) -> Instance:
        pass

    @abstractmethod
    def clone(self) -> "InternalSolver":
        """
        Returns a new copy of this solver with identical parameters, but otherwise
        completely unitialized.
        """
        pass

    @abstractmethod
    def fix(self, solution: Solution) -> None:
        """
        Fixes the values of a subset of decision variables. Missing values in the
        solution indicate variables that should be left free.
        """
        pass

    @abstractmethod
    def get_solution(self) -> Optional[Solution]:
        """
        Returns current solution found by the solver.

        If called after `solve`, returns the best primal solution found during
        the search. If called after `solve_lp`, returns the optimal solution
        to the LP relaxation. If no primal solution is available, return None.
        """
        pass

    @abstractmethod
    def get_constraint_attrs(self) -> List[str]:
        """
        Returns a list of constraint attributes supported by this solver. Used for
        testing purposes only.
        """

        pass

    @abstractmethod
    def get_constraints(
        self,
        with_static: bool = True,
        with_sa: bool = True,
        with_lhs: bool = True,
    ) -> ConstraintFeatures:
        pass

    @abstractmethod
    def get_variable_attrs(self) -> List[str]:
        """
        Returns a list of variable attributes supported by this solver. Used for
        testing purposes only.
        """
        pass

    @abstractmethod
    def get_variables(
        self,
        with_static: bool = True,
        with_sa: bool = True,
    ) -> VariableFeatures:
        """
        Returns a description of the decision variables in the problem.

        Parameters
        ----------
        with_static: bool
            If True, include features that do not change during the solution process,
            such as variable types and names. This parameter is used to reduce the
            amount of duplicated data collected by LearningSolver. Features that do
            not change are only collected once.
        with_sa: bool
            If True, collect sensitivity analysis information. For large models,
            collecting this information may be expensive, so this parameter is useful
            for reducing running times.
        """
        pass

    @abstractmethod
    def is_infeasible(self) -> bool:
        """
        Returns True if the model has been proved to be infeasible.
        Must be called after solve.
        """
        pass

    @abstractmethod
    def remove_constraints(self, names: Tuple[str, ...]) -> None:
        """
        Removes the given constraints from the model.
        """
        pass

    @abstractmethod
    def relax(self) -> None:
        """
        Drops all integrality constraints from the model.
        """
        pass

    def set_branching_priorities(self, priorities: BranchPriorities) -> None:
        """
        Sets the branching priorities for the given decision variables.

        When the MIP solver needs to decide on which variable to branch, variables
        with higher priority are picked first, given that they are fractional.
        Ties are solved arbitrarily. By default, all variables have priority zero.

        Missing values indicate variables whose priorities should not be modified.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_instance(
        self,
        instance: Instance,
        model: Any = None,
    ) -> None:
        """
        Loads the given instance into the solver.

        Parameters
        ----------
        instance: Instance
            The instance to be loaded.
        model: Any
            The concrete optimization model corresponding to this instance
            (e.g. JuMP.Model or pyomo.core.ConcreteModel). If not provided,
            it will be generated by calling `instance.to_model()`.
        """
        pass

    @abstractmethod
    def set_warm_start(self, solution: Solution) -> None:
        """
        Sets the warm start to be used by the solver.

        Only one warm start is supported. Calling this function when a warm start
        already exists will remove the previous warm start.
        """
        pass

    @abstractmethod
    def solve(
        self,
        tee: bool = False,
        iteration_cb: Optional[IterationCallback] = None,
        lazy_cb: Optional[LazyCallback] = None,
        user_cut_cb: Optional[UserCutCallback] = None,
    ) -> MIPSolveStats:
        """
        Solves the currently loaded instance. After this method finishes,
        the best solution found can be retrieved by calling `get_solution`.

        Parameters
        ----------
        iteration_cb: IterationCallback
            By default, InternalSolver makes a single call to the native `solve`
            method and returns the result. If an iteration callback is provided
            instead, InternalSolver enters a loop, where `solve` and `iteration_cb`
            are called alternatively. To stop the loop, `iteration_cb` should return
            False. Any other result causes the solver to loop again.
        lazy_cb: LazyCallback
            This function is called whenever the solver finds a new candidate
            solution and can be used to add lazy constraints to the model. Only the
            following operations within the callback are allowed:
                - Querying the value of a variable
                - Querying if a constraint is satisfied
                - Adding a new constraint to the problem
            Additional operations may be allowed by specific subclasses.
        user_cut_cb: UserCutCallback
            This function is called whenever the solver found a new integer-infeasible
            solution and needs to generate cutting planes to cut it off.
        tee: bool
            If true, prints the solver log to the screen.
        """
        pass

    @abstractmethod
    def solve_lp(
        self,
        tee: bool = False,
    ) -> LPSolveStats:
        """
        Solves the LP relaxation of the currently loaded instance. After this
        method finishes, the solution can be retrieved by calling `get_solution`.

        This method should not permanently modify the problem. That is, subsequent
        calls to `solve` should solve the original MIP, not the LP relaxation.

        Parameters
        ----------
        tee
            If true, prints the solver log to the screen.
        """
        pass
