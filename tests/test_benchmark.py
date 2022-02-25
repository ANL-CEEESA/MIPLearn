#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import os.path

from scipy.stats import randint

from miplearn.benchmark import BenchmarkRunner
from miplearn.problems.stab import (
    MaxWeightStableSetInstance,
    MaxWeightStableSetGenerator,
)
from miplearn.solvers.learning import LearningSolver


def test_benchmark() -> None:
    for n_jobs in [1, 4]:
        # Generate training and test instances
        generator = MaxWeightStableSetGenerator(n=randint(low=25, high=26))
        train_instances = [
            MaxWeightStableSetInstance(data.graph, data.weights)
            for data in generator.generate(5)
        ]
        test_instances = [
            MaxWeightStableSetInstance(data.graph, data.weights)
            for data in generator.generate(3)
        ]

        # Solve training instances
        training_solver = LearningSolver()
        training_solver.parallel_solve(train_instances, n_jobs=n_jobs)  # type: ignore

        # Benchmark
        test_solvers = {
            "Strategy A": LearningSolver(),
            "Strategy B": LearningSolver(),
        }
        benchmark = BenchmarkRunner(test_solvers)
        benchmark.fit(train_instances, n_jobs=n_jobs)  # type: ignore
        benchmark.parallel_solve(
            test_instances,  # type: ignore
            n_jobs=n_jobs,
            n_trials=2,
        )
        benchmark.write_csv("/tmp/benchmark.csv")
        assert os.path.isfile("/tmp/benchmark.csv")
        assert benchmark.results.values.shape == (12, 21)
