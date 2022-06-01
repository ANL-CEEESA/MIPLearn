#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import os
from typing import Dict, List, Any, Optional, Callable

import pandas as pd

from miplearn.components.component import Component
from miplearn.instance.base import Instance
from miplearn.solvers.learning import LearningSolver, FileInstanceWrapper
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Utility class that simplifies the task of comparing the performance of different
    solvers.

    Parameters
    ----------
    solvers: Dict[str, LearningSolver]
        Dictionary containing the solvers to compare. Solvers may have different
        arguments and components. The key should be the name of the solver. It
        appears in the exported tables of results.
    """

    def __init__(self, solvers: Dict[str, LearningSolver]) -> None:
        self.solvers: Dict[str, LearningSolver] = solvers
        self.results = pd.DataFrame(
            columns=[
                "Solver",
                "Instance",
            ]
        )

    def parallel_solve(
        self,
        filenames: List[str],
        build_model: Callable,
        n_jobs: int = 1,
        n_trials: int = 1,
        progress: bool = False,
    ) -> None:
        self._silence_miplearn_logger()
        trials = filenames * n_trials
        for (solver_name, solver) in self.solvers.items():
            results = solver.parallel_solve(
                trials,
                build_model,
                n_jobs=n_jobs,
                label="benchmark (%s)" % solver_name,
                progress=progress,
            )
            for i in range(len(trials)):
                idx = i % len(filenames)
                results[i]["Solver"] = solver_name
                results[i]["Instance"] = idx
                self.results = self.results.append(pd.DataFrame([results[i]]))
        self._restore_miplearn_logger()

    def write_csv(self, filename: str) -> None:
        """
        Writes the collected results to a CSV file.

        Parameters
        ----------
        filename: str
            The name of the file.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.results.to_csv(filename)

    def fit(
        self,
        filenames: List[str],
        build_model: Callable,
        progress: bool = False,
        n_jobs: int = 1,
    ) -> None:
        components = []
        instances: List[Instance] = [
            FileInstanceWrapper(f, build_model, mode="r") for f in filenames
        ]
        for (solver_name, solver) in self.solvers.items():
            if solver_name == "baseline":
                continue
            components += solver.components.values()
        Component.fit_multiple(
            components,
            instances,
            n_jobs=n_jobs,
            progress=progress,
        )

    def _silence_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        self.prev_log_level = miplearn_logger.getEffectiveLevel()
        miplearn_logger.setLevel(logging.WARNING)

    def _restore_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        miplearn_logger.setLevel(self.prev_log_level)

    def write_svg(
        self,
        output: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        sns.set_style("whitegrid")
        sns.set_palette("Blues_r")
        groups = self.results.groupby("Instance")
        best_lower_bound = groups["mip_lower_bound"].transform("max")
        best_upper_bound = groups["mip_upper_bound"].transform("min")
        self.results["Relative lower bound"] = self.results["mip_lower_bound"] / best_lower_bound
        self.results["Relative upper bound"] = self.results["mip_upper_bound"] / best_upper_bound

        if (self.results["mip_sense"] == "min").any():
            primal_column = "Relative upper bound"
            obj_column = "mip_upper_bound"
            predicted_obj_column = "Objective: Predicted upper bound"
        else:
            primal_column = "Relative lower bound"
            obj_column = "mip_lower_bound"
            predicted_obj_column = "Objective: Predicted lower bound"

        palette = {
            "baseline": "#9b59b6",
            "ml-exact": "#3498db",
            "ml-heuristic": "#95a5a6",
        }
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(8, 8),
        )

        # Wallclock time
        sns.stripplot(
            x="Solver",
            y="mip_wallclock_time",
            data=self.results,
            ax=ax1,
            jitter=0.25,
            palette=palette,
            size=2.0,
        )
        sns.barplot(
            x="Solver",
            y="mip_wallclock_time",
            data=self.results,
            ax=ax1,
            errwidth=0.0,
            alpha=0.4,
            palette=palette,
        )
        ax1.set(ylabel="Wallclock time (s)")

        # Gap
        sns.stripplot(
            x="Solver",
            y="Gap",
            jitter=0.25,
            data=self.results[self.results["Solver"] != "ml-heuristic"],
            ax=ax2,
            palette=palette,
            size=2.0,
        )
        ax2.set(ylabel="Relative MIP gap")

        # Relative primal bound
        sns.stripplot(
            x="Solver",
            y=primal_column,
            jitter=0.25,
            data=self.results[self.results["Solver"] == "ml-heuristic"],
            ax=ax3,
            palette=palette,
            size=2.0,
        )
        sns.scatterplot(
            x=obj_column,
            y=predicted_obj_column,
            hue="Solver",
            data=self.results[self.results["Solver"] == "ml-exact"],
            ax=ax4,
            palette=palette,
            size=2.0,
        )

        # Predicted vs actual primal bound
        xlim, ylim = ax4.get_xlim(), ax4.get_ylim()
        ax4.plot(
            [-1e10, 1e10],
            [-1e10, 1e10],
            ls="-",
            color="#cccccc",
        )
        ax4.set_xlim(xlim)
        ax4.set_ylim(xlim)
        ax4.get_legend().remove()
        ax4.set(
            ylabel="Predicted optimal value",
            xlabel="Actual optimal value",
        )

        fig.tight_layout()
        plt.savefig(output)


@ignore_warnings(category=ConvergenceWarning)
def run_benchmarks(
    train_instances: List[Instance],
    test_instances: List[Instance],
    n_jobs: int = 4,
    n_trials: int = 1,
    progress: bool = False,
    solver: Any = None,
) -> None:
    if solver is None:
        solver = GurobiPyomoSolver()
    benchmark = BenchmarkRunner(
        solvers={
            "baseline": LearningSolver(
                solver=solver.clone(),
            ),
            "ml-exact": LearningSolver(
                solver=solver.clone(),
            ),
            "ml-heuristic": LearningSolver(
                solver=solver.clone(),
                mode="heuristic",
            ),
        }
    )
    benchmark.solvers["baseline"].parallel_solve(
        train_instances,
        n_jobs=n_jobs,
        progress=progress,
    )
    benchmark.fit(
        train_instances,
        n_jobs=n_jobs,
        progress=progress,
    )
    benchmark.parallel_solve(
        test_instances,
        n_jobs=n_jobs,
        n_trials=n_trials,
        progress=progress,
    )
    plot(benchmark.results)


