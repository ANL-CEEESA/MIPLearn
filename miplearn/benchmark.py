#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from copy import deepcopy

import pandas as pd
import numpy as np
import logging
from tqdm.auto import tqdm

from .solvers.learning import LearningSolver


class BenchmarkRunner:
    def __init__(self, solvers):
        assert isinstance(solvers, dict)
        for solver in solvers.values():
            assert isinstance(solver, LearningSolver)
        self.solvers = solvers
        self.results = None

    def solve(self, instances, tee=False):
        for (solver_name, solver) in self.solvers.items():
            for i in tqdm(range(len((instances)))):
                results = solver.solve(deepcopy(instances[i]), tee=tee)
                self._push_result(
                    results,
                    solver=solver,
                    solver_name=solver_name,
                    instance=i,
                )

    def parallel_solve(
        self,
        instances,
        n_jobs=1,
        n_trials=1,
        index_offset=0,
    ):
        self._silence_miplearn_logger()
        trials = instances * n_trials
        for (solver_name, solver) in self.solvers.items():
            results = solver.parallel_solve(
                trials,
                n_jobs=n_jobs,
                label="Solve (%s)" % solver_name,
                output=None,
            )
            for i in range(len(trials)):
                idx = (i % len(instances)) + index_offset
                self._push_result(
                    results[i],
                    solver=solver,
                    solver_name=solver_name,
                    instance=idx,
                )
        self._restore_miplearn_logger()

    def raw_results(self):
        return self.results

    def save_results(self, filename):
        self.results.to_csv(filename)

    def load_results(self, filename):
        self.results = pd.read_csv(filename, index_col=0)

    def load_state(self, filename):
        for (solver_name, solver) in self.solvers.items():
            solver.load_state(filename)

    def fit(self, training_instances):
        for (solver_name, solver) in self.solvers.items():
            solver.fit(training_instances)

    def _push_result(self, result, solver, solver_name, instance):
        if self.results is None:
            self.results = pd.DataFrame(
                # Show the following columns first in the CSV file
                columns=[
                    "Solver",
                    "Instance",
                ]
            )

        lb = result["Lower bound"]
        ub = result["Upper bound"]
        result["Solver"] = solver_name
        result["Instance"] = instance
        result["Gap"] = (ub - lb) / lb
        result["Mode"] = solver.mode
        self.results = self.results.append(pd.DataFrame([result]))

        # Compute relative statistics
        groups = self.results.groupby("Instance")
        best_lower_bound = groups["Lower bound"].transform("max")
        best_upper_bound = groups["Upper bound"].transform("min")
        best_gap = groups["Gap"].transform("min")
        best_nodes = np.maximum(1, groups["Nodes"].transform("min"))
        best_wallclock_time = groups["Wallclock time"].transform("min")
        self.results["Relative lower bound"] = (
            self.results["Lower bound"] / best_lower_bound
        )
        self.results["Relative upper bound"] = (
            self.results["Upper bound"] / best_upper_bound
        )
        self.results["Relative wallclock time"] = (
            self.results["Wallclock time"] / best_wallclock_time
        )
        self.results["Relative Gap"] = self.results["Gap"] / best_gap
        self.results["Relative Nodes"] = self.results["Nodes"] / best_nodes

    def save_chart(self, filename):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from numpy import median

        sns.set_style("whitegrid")
        sns.set_palette("Blues_r")
        results = self.raw_results()
        results["Gap (%)"] = results["Gap"] * 100.0

        sense = results.loc[0, "Sense"]
        if sense == "min":
            primal_column = "Relative upper bound"
            obj_column = "Upper bound"
            predicted_obj_column = "Predicted UB"
        else:
            primal_column = "Relative lower bound"
            obj_column = "Lower bound"
            predicted_obj_column = "Predicted LB"

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nrows=1,
            ncols=4,
            figsize=(12, 4),
            gridspec_kw={"width_ratios": [2, 1, 1, 2]},
        )

        # Figure 1: Solver x Wallclock time
        sns.stripplot(
            x="Solver",
            y="Wallclock time",
            data=results,
            ax=ax1,
            jitter=0.25,
            size=4.0,
        )
        sns.barplot(
            x="Solver",
            y="Wallclock time",
            data=results,
            ax=ax1,
            errwidth=0.0,
            alpha=0.4,
            estimator=median,
        )
        ax1.set(ylabel="Wallclock time (s)")

        # Figure 2: Solver x Gap (%)
        ax2.set_ylim(-0.5, 5.5)
        sns.stripplot(
            x="Solver",
            y="Gap (%)",
            jitter=0.25,
            data=results[results["Mode"] != "heuristic"],
            ax=ax2,
            size=4.0,
        )

        # Figure 3: Solver x Primal Value
        ax3.set_ylim(0.95, 1.05)
        sns.stripplot(
            x="Solver",
            y=primal_column,
            jitter=0.25,
            data=results[results["Mode"] == "heuristic"],
            ax=ax3,
        )

        # Figure 4: Predicted vs Actual Objective Value
        sns.scatterplot(
            x=obj_column,
            y=predicted_obj_column,
            hue="Solver",
            data=results[results["Mode"] != "heuristic"],
            ax=ax4,
        )
        xlim, ylim = ax4.get_xlim(), ax4.get_ylim()
        ax4.plot([-1e10, 1e10], [-1e10, 1e10], ls="-", color="#cccccc")
        ax4.set_xlim(xlim)
        ax4.set_ylim(ylim)
        ax4.get_legend().remove()

        fig.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=150)

    def _silence_miplearn_logger(self):
        miplearn_logger = logging.getLogger("miplearn")
        self.prev_log_level = miplearn_logger.getEffectiveLevel()
        miplearn_logger.setLevel(logging.WARNING)

    def _restore_miplearn_logger(self):
        miplearn_logger = logging.getLogger("miplearn")
        miplearn_logger.setLevel(self.prev_log_level)
