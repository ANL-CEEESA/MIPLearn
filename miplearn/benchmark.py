#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import os
from typing import Dict, Union, List

import pandas as pd

from miplearn.instance import Instance
from miplearn.solvers.learning import LearningSolver

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Utility class that simplifies the task of comparing the performance of different
    solvers.

    Example
    -------
    ```python
    benchmark = BenchmarkRunner({
        "Baseline": LearningSolver(...),
        "Strategy A": LearningSolver(...),
        "Strategy B": LearningSolver(...),
        "Strategy C": LearningSolver(...),
    })
    benchmark.fit(train_instances)
    benchmark.parallel_solve(test_instances, n_jobs=5)
    benchmark.save_results("result.csv")
    ```

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
        instances: List[Instance],
        n_jobs: int = 1,
        n_trials: int = 3,
    ) -> None:
        """
        Solves the given instances in parallel and collect benchmark statistics.

        Parameters
        ----------
        instances: List[Instance]
            List of instances to solve. This can either be a list of instances
            already loaded in memory, or a list of filenames pointing to pickled (and
            optionally gzipped) files.
        n_jobs: int
            List of instances to solve in parallel at a time.
        n_trials: int
            How many times each instance should be solved.
        """
        self._silence_miplearn_logger()
        trials = instances * n_trials
        for (solver_name, solver) in self.solvers.items():
            results = solver.parallel_solve(
                trials,
                n_jobs=n_jobs,
                label="Solve (%s)" % solver_name,
                discard_outputs=True,
            )
            for i in range(len(trials)):
                idx = i % len(instances)
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

    def fit(self, instances: List[Instance]) -> None:
        """
        Trains all solvers with the provided training instances.

        Parameters
        ----------
        instances:  List[Instance]
            List of training instances.
        """
        for (solver_name, solver) in self.solvers.items():
            logger.debug(f"Fitting {solver_name}...")
            solver.fit(instances)

    def _silence_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        self.prev_log_level = miplearn_logger.getEffectiveLevel()
        miplearn_logger.setLevel(logging.WARNING)

    def _restore_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        miplearn_logger.setLevel(self.prev_log_level)
