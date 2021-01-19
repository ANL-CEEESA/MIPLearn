#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from copy import deepcopy

import pandas as pd
import numpy as np
import logging
from tqdm.auto import tqdm
import os

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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.results.to_csv(filename)

    def load_results(self, filename):
        self.results = pd.concat([self.results, pd.read_csv(filename, index_col=0)])

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

    def _silence_miplearn_logger(self):
        miplearn_logger = logging.getLogger("miplearn")
        self.prev_log_level = miplearn_logger.getEffectiveLevel()
        miplearn_logger.setLevel(logging.WARNING)

    def _restore_miplearn_logger(self):
        miplearn_logger = logging.getLogger("miplearn")
        miplearn_logger.setLevel(self.prev_log_level)
