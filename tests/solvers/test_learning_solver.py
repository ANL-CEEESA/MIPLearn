#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import os
import tempfile
from typing import List, cast

import dill

from miplearn.instance.base import Instance
from miplearn.instance.picklegz import PickleGzInstance, write_pickle_gz, read_pickle_gz
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.learning import LearningSolver

# noinspection PyUnresolvedReferences
from tests.solvers.test_internal_solver import internal_solvers
from miplearn.solvers.tests import assert_equals

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

            solver.solve(instance)
            assert len(instance.get_samples()) > 0
            sample = instance.get_samples()[0]

            assert_equals(
                sample.get_vector("mip_var_values"), [1.0, 0.0, 1.0, 1.0, 61.0]
            )
            assert sample.get_scalar("mip_lower_bound") == 1183.0
            assert sample.get_scalar("mip_upper_bound") == 1183.0
            mip_log = sample.get_scalar("mip_log")
            assert mip_log is not None
            assert len(mip_log) > 100

            assert_equals(
                sample.get_vector("lp_var_values"), [1.0, 0.923077, 1.0, 0.0, 67.0]
            )
            assert_equals(sample.get_scalar("lp_value"), 1287.923077)
            lp_log = sample.get_scalar("lp_log")
            assert lp_log is not None
            assert len(lp_log) > 100

            solver.fit([instance], n_jobs=4)
            solver.solve(instance)

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
        solver.solve(instance)
        solver.fit([instance])
        solver.solve(instance)


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
        solver.solve(instances[0])
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


def test_simulate_perfect() -> None:
    internal_solver = GurobiSolver()
    instance = internal_solver.build_test_instance_knapsack()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        write_pickle_gz(instance, tmp.name)
        solver = LearningSolver(
            solver=internal_solver,
            simulate_perfect=True,
        )
        stats = solver.solve(PickleGzInstance(tmp.name))
        assert stats["mip_lower_bound"] == stats["Objective: Predicted lower bound"]


def test_gap() -> None:
    assert LearningSolver._compute_gap(ub=0.0, lb=0.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=0.5) == 0.5
    assert LearningSolver._compute_gap(ub=1.0, lb=1.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=-1.0) is None
    assert LearningSolver._compute_gap(ub=1.0, lb=None) is None
    assert LearningSolver._compute_gap(ub=None, lb=1.0) is None
    assert LearningSolver._compute_gap(ub=None, lb=None) is None
