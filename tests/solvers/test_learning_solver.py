#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import os
import tempfile
from os.path import exists
from typing import List, cast

import dill
from scipy.stats import randint

from miplearn.features.sample import Hdf5Sample
from miplearn.instance.base import Instance
from miplearn.instance.picklegz import (
    PickleGzInstance,
    write_pickle_gz,
    read_pickle_gz,
    save,
)
from miplearn.problems.stab import MaxWeightStableSetGenerator, build_stab_model
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.learning import LearningSolver
from miplearn.solvers.tests import assert_equals

# noinspection PyUnresolvedReferences
from tests.solvers.test_internal_solver import internal_solvers

logger = logging.getLogger(__name__)


def test_learning_solver(
    internal_solvers: List[InternalSolver],
) -> None:
    for mode in ["exact", "heuristic"]:
        for internal_solver in internal_solvers:
            logger.info("Solver: %s" % internal_solver)
            instance = internal_solver.build_test_instance_knapsack()
            solver = LearningSolver(
                solver=internal_solver,
                mode=mode,
            )

            solver._solve(instance)
            assert len(instance.get_samples()) > 0
            sample = instance.get_samples()[0]

            assert_equals(
                sample.get_array("mip_var_values"), [1.0, 0.0, 1.0, 1.0, 61.0]
            )
            assert sample.get_scalar("mip_lower_bound") == 1183.0
            assert sample.get_scalar("mip_upper_bound") == 1183.0
            mip_log = sample.get_scalar("mip_log")
            assert mip_log is not None
            assert len(mip_log) > 100

            assert_equals(
                sample.get_array("lp_var_values"), [1.0, 0.923077, 1.0, 0.0, 67.0]
            )
            assert_equals(sample.get_scalar("lp_value"), 1287.923077)
            lp_log = sample.get_scalar("lp_log")
            assert lp_log is not None
            assert len(lp_log) > 100

            solver._fit([instance], n_jobs=4)
            solver._solve(instance)

            # Assert solver is picklable
            with tempfile.TemporaryFile() as file:
                dill.dump(solver, file)


def test_solve_without_lp(
    internal_solvers: List[InternalSolver],
) -> None:
    for internal_solver in internal_solvers:
        logger.info("Solver: %s" % internal_solver)
        instance = internal_solver.build_test_instance_knapsack()
        solver = LearningSolver(
            solver=internal_solver,
            solve_lp=False,
        )
        solver._solve(instance)
        solver._fit([instance])
        solver._solve(instance)


def test_parallel_solve(
    internal_solvers: List[InternalSolver],
) -> None:
    for internal_solver in internal_solvers:
        instances = [internal_solver.build_test_instance_knapsack() for _ in range(10)]
        solver = LearningSolver(solver=internal_solver)
        results = solver.parallel_solve(instances, n_jobs=3)
        assert len(results) == 10
        for instance in instances:
            assert len(instance.get_samples()) == 1


def test_solve_fit_from_disk(
    internal_solvers: List[InternalSolver],
) -> None:
    for internal_solver in internal_solvers:
        # Create instances and pickle them
        instances: List[Instance] = []
        for k in range(3):
            instance = internal_solver.build_test_instance_knapsack()
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
                instances += [PickleGzInstance(file.name)]
                write_pickle_gz(instance, file.name)

        # Test: solve
        solver = LearningSolver(solver=internal_solver)
        solver._solve(instances[0])
        instance_loaded = read_pickle_gz(cast(PickleGzInstance, instances[0]).filename)
        assert len(instance_loaded.get_samples()) > 0

        # Test: parallel_solve
        solver.parallel_solve(instances)
        for instance in instances:
            instance_loaded = read_pickle_gz(cast(PickleGzInstance, instance).filename)
            assert len(instance_loaded.get_samples()) > 0

        # Delete temporary files
        for instance in instances:
            os.remove(cast(PickleGzInstance, instance).filename)


def test_basic_usage() -> None:
    with tempfile.TemporaryDirectory() as dirname:
        # Generate instances
        data = MaxWeightStableSetGenerator(n=randint(low=20, high=21)).generate(4)
        train_files = save(data[0:3], f"{dirname}/train")
        test_files = save(data[3:4], f"{dirname}/test")

        # Solve training instances
        solver = LearningSolver()
        stats = solver.solve(train_files, build_stab_model)
        assert len(stats) == 3
        for f in train_files:
            sample_filename = f.replace(".pkl.gz", ".h5")
            assert exists(sample_filename)
            sample = Hdf5Sample(sample_filename)
            assert sample.get_scalar("mip_lower_bound") > 0

        # Fit
        solver.fit(train_files, build_stab_model)

        # Solve test instances
        stats = solver.solve(test_files, build_stab_model)
        assert isinstance(stats, list)
        assert "Objective: Predicted lower bound" in stats[0].keys()


def test_gap() -> None:
    assert LearningSolver._compute_gap(ub=0.0, lb=0.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=0.5) == 0.5
    assert LearningSolver._compute_gap(ub=1.0, lb=1.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=-1.0) is None
    assert LearningSolver._compute_gap(ub=1.0, lb=None) is None
    assert LearningSolver._compute_gap(ub=None, lb=1.0) is None
    assert LearningSolver._compute_gap(ub=None, lb=None) is None
