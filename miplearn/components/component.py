#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from typing import Any, List, Union, TYPE_CHECKING, Tuple, Dict, Optional, Hashable

from miplearn.extractors import InstanceIterator
from miplearn.instance import Instance
from miplearn.types import LearningSolveStats, TrainingSample, Features

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


# noinspection PyMethodMayBeStatic
class Component:
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
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        """
        Method called by LearningSolver before the root LP relaxation is solved.

        Parameters
        ----------
        solver: LearningSolver
            The solver calling this method.
        instance: Instance
            The instance being solved.
        model
            The concrete optimization model being solved.
        stats: LearningSolveStats
            A dictionary containing statistics about the solution process, such as
            number of nodes explored and running time. Components are free to add
            their own statistics here. For example, PrimalSolutionComponent adds
            statistics regarding the number of predicted variables. All statistics in
            this dictionary are exported to the benchmark CSV file.
        features: Features
            Features describing the model.
        training_data: TrainingSample
            A dictionary containing data that may be useful for training machine
            learning models and accelerating the solution process. Components are
            free to add their own training data here. For example,
            PrimalSolutionComponent adds the current primal solution. The data must
            be pickable.
        """
        return

    def after_solve_lp(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        """
        Method called by LearningSolver after the root LP relaxation is solved.
        See before_solve_lp for a description of the parameters.
        """
        return

    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        """
        Method called by LearningSolver before the MIP is solved.
        See before_solve_lp for a description of the parameters.
        """
        return

    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        """
        Method called by LearningSolver after the MIP is solved.
        See before_solve_lp for a description of the parameters.
        """
        return

    @staticmethod
    def sample_xy(
        features: Features,
        sample: TrainingSample,
    ) -> Tuple[Dict, Dict]:
        """
        Given a set of features and a training sample, returns a pair of x and y
        dictionaries containing, respectively, the matrices of ML features and the
        labels for the sample. If the training sample does not include label
        information, returns (x, {}).
        """
        pass

    def xy_instances(
        self,
        instances: Union[List[str], List[Instance]],
    ) -> Tuple[Dict, Dict]:
        x_combined: Dict = {}
        y_combined: Dict = {}
        for instance in InstanceIterator(instances):
            assert isinstance(instance, Instance)
            for sample in instance.training_data:
                xy = self.sample_xy(instance.features, sample)
                if xy is None:
                    continue
                x_sample, y_sample = xy
                for cat in x_sample.keys():
                    if cat not in x_combined:
                        x_combined[cat] = []
                        y_combined[cat] = []
                    x_combined[cat] += x_sample[cat]
                    y_combined[cat] += y_sample[cat]
        return x_combined, y_combined

    def fit(
        self,
        training_instances: Union[List[str], List[Instance]],
    ) -> None:
        x, y = self.xy_instances(training_instances)
        for cat in x.keys():
            x[cat] = np.array(x[cat])
            y[cat] = np.array(y[cat])
        self.fit_xy(x, y)

    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        """
        Given two dictionaries x and y, mapping the name of the category to matrices
        of features and targets, this function does two things. First, for each
        category, it creates a clone of the prototype regressor/classifier. Second,
        it passes (x[category], y[category]) to the clone's fit method.
        """
        return

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

    def evaluate(self, instances: Union[List[str], List[Instance]]) -> List:
        ev = []
        for instance in InstanceIterator(instances):
            for sample in instance.training_data:
                ev += [self.sample_evaluate(instance.features, sample)]
        return ev

    def sample_evaluate(
        self,
        features: Features,
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        return {}
