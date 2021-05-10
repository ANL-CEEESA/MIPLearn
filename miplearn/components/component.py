#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any, List, TYPE_CHECKING, Tuple, Dict, Hashable, Optional

import numpy as np
from p_tqdm import p_umap

from miplearn.features import Sample
from miplearn.instance.base import Instance
from miplearn.types import LearningSolveStats

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

    def after_solve_lp(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        """
        Method called by LearningSolver after the root LP relaxation is solved.
        See before_solve_lp for a description of the parameters.
        """
        return

    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        """
        Method called by LearningSolver after the MIP is solved.
        See before_solve_lp for a description of the parameters.
        """
        return

    def before_solve_lp(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
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
        sample: miplearn.features.Sample
            An object containing data that may be useful for training machine
            learning models and accelerating the solution process. Components are
            free to add their own training data here.
        """
        return

    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        """
        Method called by LearningSolver before the MIP is solved.
        See before_solve_lp for a description of the parameters.
        """
        return

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

    def sample_evaluate(
        self,
        instance: Optional[Instance],
        sample: Sample,
    ) -> Dict[Hashable, Dict[str, float]]:
        return {}

    def sample_xy(
        self,
        instance: Optional[Instance],
        sample: Sample,
    ) -> Tuple[Dict, Dict]:
        """
        Returns a pair of x and y dictionaries containing, respectively, the matrices
        of ML features and the labels for the sample. If the training sample does not
        include label information, returns (x, {}).
        """
        pass

    def pre_fit(self, pre: List[Any]) -> None:
        pass

    def user_cut_cb(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> None:
        return

    def pre_sample_xy(self, instance: Instance, sample: Sample) -> Any:
        pass

    @staticmethod
    def fit_multiple(
        components: Dict[str, "Component"],
        instances: List[Instance],
        n_jobs: int = 1,
    ) -> None:
        def _pre_sample_xy(instance: Instance) -> Dict:
            pre_instance: Dict = {}
            for (cname, comp) in components.items():
                pre_instance[cname] = []
            instance.load()
            for sample in instance.samples:
                for (cname, comp) in components.items():
                    pre_instance[cname].append(comp.pre_sample_xy(instance, sample))
            instance.free()
            return pre_instance

        pre = p_umap(_pre_sample_xy, instances, num_cpus=n_jobs)
        pre_combined: Dict = {}
        for (cname, comp) in components.items():
            pre_combined[cname] = []
            for p in pre:
                pre_combined[cname].extend(p[cname])

        for (cname, comp) in components.items():
            comp.pre_fit(pre_combined[cname])

        def _sample_xy(instance: Instance) -> Tuple[Dict, Dict]:
            x_instance: Dict = {}
            y_instance: Dict = {}
            for (cname, comp) in components.items():
                x_instance[cname] = {}
                y_instance[cname] = {}
            instance.load()
            for sample in instance.samples:
                for (cname, comp) in components.items():
                    x = x_instance[cname]
                    y = y_instance[cname]
                    x_sample, y_sample = comp.sample_xy(instance, sample)
                    for cat in x_sample.keys():
                        if cat not in x:
                            x[cat] = []
                            y[cat] = []
                        x[cat] += x_sample[cat]
                        y[cat] += y_sample[cat]
            instance.free()
            return x_instance, y_instance

        xy = p_umap(_sample_xy, instances)

        for (cname, comp) in components.items():
            x_comp: Dict = {}
            y_comp: Dict = {}
            for (x, y) in xy:
                for cat in x[cname].keys():
                    if cat not in x_comp:
                        x_comp[cat] = []
                        y_comp[cat] = []
                    x_comp[cat].extend(x[cname][cat])
                    y_comp[cat].extend(y[cname][cat])
            for cat in x_comp.keys():
                x_comp[cat] = np.array(x_comp[cat], dtype=np.float32)
                y_comp[cat] = np.array(y_comp[cat])
            comp.fit_xy(x_comp, y_comp)
