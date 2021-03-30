#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod
from typing import Any, List, Union, TYPE_CHECKING, Tuple, Dict

from miplearn.instance import Instance
from miplearn.types import LearningSolveStats, TrainingSample

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


# noinspection PyMethodMayBeStatic
class Component(ABC):
    """
    A Component is an object which adds functionality to a LearningSolver.

    For better code maintainability, LearningSolver simply delegates most of its
    functionality to Components. Each Component is responsible for exactly one ML
    strategy.
    """

    def before_solve_lp(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> None:
        """
        Method called by LearningSolver before the root LP relaxation is solved.

        Parameters
        ----------
        solver
            The solver calling this method.
        instance
            The instance being solved.
        model
            The concrete optimization model being solved.
        """
        return

    def after_solve_lp(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        training_data: TrainingSample,
    ) -> None:
        """
        Method called by LearningSolver after the root LP relaxation is solved.

        Parameters
        ----------
        solver: LearningSolver
            The solver calling this method.
        instance: Instance
            The instance being solved.
        model: Any
            The concrete optimization model being solved.
        stats: LearningSolveStats
            A dictionary containing statistics about the solution process, such as
            number of nodes explored and running time. Components are free to add
            their own statistics here. For example, PrimalSolutionComponent adds
            statistics regarding the number of predicted variables. All statistics in
            this dictionary are exported to the benchmark CSV file.
        training_data: TrainingSample
            A dictionary containing data that may be useful for training machine
            learning models and accelerating the solution process. Components are
            free to add their own training data here. For example,
            PrimalSolutionComponent adds the current primal solution. The data must
            be pickable.
        """
        return

    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> None:
        """
        Method called by LearningSolver before the MIP is solved.

        Parameters
        ----------
        solver
            The solver calling this method.
        instance
            The instance being solved.
        model
            The concrete optimization model being solved.
        """
        return

    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        training_data: TrainingSample,
    ) -> None:
        """
        Method called by LearningSolver after the MIP is solved.

        Parameters
        ----------
        solver: LearningSolver
            The solver calling this method.
        instance: Instance
            The instance being solved.
        model: Any
            The concrete optimization model being solved.
        stats: LearningSolveStats
            A dictionary containing statistics about the solution process, such as
            number of nodes explored and running time. Components are free to add
            their own statistics here. For example, PrimalSolutionComponent adds
            statistics regarding the number of predicted variables. All statistics in
            this dictionary are exported to the benchmark CSV file.
        training_data: TrainingSample
            A dictionary containing data that may be useful for training machine
            learning models and accelerating the solution process. Components are
            free to add their own training data here. For example,
            PrimalSolutionComponent adds the current primal solution. The data must
            be pickable.
        """
        return

    def fit(
        self,
        training_instances: Union[List[str], List[Instance]],
    ) -> None:
        return

    def xy(
        self,
        instance: Any,
        training_sample: TrainingSample,
    ) -> Tuple[Dict, Dict]:
        """
        Given a training sample, returns a pair of x and y dictionaries containing,
        respectively, the matrices of ML features and the labels for the sample.
        """
        return {}, {}

    def iteration_cb(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> bool:
        """
        Method called by LearningSolver at the end of each iteration.

        After solving the MIP, LearningSolver calls `iteration_cb` of each component,
        giving them a chance to modify the problem and resolve it before the solution
        process ends. For example, the lazy constraint component uses `iteration_cb`
        to check that all lazy constraints are satisfied.

        If `iteration_cb` returns False for all components, the solution process
        ends. If it retunrs True for any component, the MIP is solved again.

        Parameters
        ----------
        solver: LearningSolver
            The solver calling this method.
        instance: Instance
            The instance being solved.
        model: Any
            The concrete optimization model being solved.
        """
        return False

    def lazy_cb(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> None:
        return
