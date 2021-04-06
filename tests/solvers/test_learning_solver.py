#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import dill
import pickle
import tempfile
import os

from miplearn.instance import PickleGzInstance, write_pickle_gz, read_pickle_gz
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.learning import LearningSolver
from . import _get_knapsack_instance, get_internal_solvers

logger = logging.getLogger(__name__)


def test_learning_solver():
    for mode in ["exact", "heuristic"]:
        for internal_solver in get_internal_solvers():
            logger.info("Solver: %s" % internal_solver)
            instance = _get_knapsack_instance(internal_solver)
            solver = LearningSolver(
                solver=internal_solver,
                mode=mode,
            )

            solver.solve(instance)
            assert hasattr(instance, "features")

            sample = instance.training_data[0]
            assert sample.solution["x"][0] == 1.0
            assert sample.solution["x"][1] == 0.0
            assert sample.solution["x"][2] == 1.0
            assert sample.solution["x"][3] == 1.0
            assert sample.lower_bound == 1183.0
            assert sample.upper_bound == 1183.0
            assert round(sample.lp_solution["x"][0], 3) == 1.000
            assert round(sample.lp_solution["x"][1], 3) == 0.923
            assert round(sample.lp_solution["x"][2], 3) == 1.000
            assert round(sample.lp_solution["x"][3], 3) == 0.000
            assert round(sample.lp_value, 3) == 1287.923
            assert len(sample.mip_log) > 100

            solver.fit([instance])
            solver.solve(instance)

            # Assert solver is picklable
            with tempfile.TemporaryFile() as file:
                dill.dump(solver, file)


def test_solve_without_lp():
    for internal_solver in get_internal_solvers():
        logger.info("Solver: %s" % internal_solver)
        instance = _get_knapsack_instance(internal_solver)
        solver = LearningSolver(
            solver=internal_solver,
            solve_lp=False,
        )
        solver.solve(instance)
        solver.fit([instance])
        solver.solve(instance)


def test_parallel_solve():
    for internal_solver in get_internal_solvers():
        instances = [_get_knapsack_instance(internal_solver) for _ in range(10)]
        solver = LearningSolver(solver=internal_solver)
        results = solver.parallel_solve(instances, n_jobs=3)
        assert len(results) == 10
        for instance in instances:
            data = instance.training_data[0]
            assert len(data.solution["x"].keys()) == 4


def test_solve_fit_from_disk():
    for internal_solver in get_internal_solvers():
        # Create instances and pickle them
        instances = []
        for k in range(3):
            instance = _get_knapsack_instance(internal_solver)
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
                instances += [PickleGzInstance(file.name)]
                write_pickle_gz(instance, file.name)

        # Test: solve
        solver = LearningSolver(solver=internal_solver)
        solver.solve(instances[0])
        instance_loaded = read_pickle_gz(instances[0].filename)
        assert len(instance_loaded.training_data) > 0
        assert instance_loaded.features.instance is not None
        assert instance_loaded.features.variables is not None
        assert instance_loaded.features.constraints is not None

        # Test: parallel_solve
        solver.parallel_solve(instances)
        for instance in instances:
            instance_loaded = read_pickle_gz(instance.filename)
            assert len(instance_loaded.training_data) > 0
            assert instance_loaded.features.instance is not None
            assert instance_loaded.features.variables is not None
            assert instance_loaded.features.constraints is not None

        # Delete temporary files
        for instance in instances:
            os.remove(instance.filename)


def test_simulate_perfect():
    internal_solver = GurobiSolver
    instance = _get_knapsack_instance(internal_solver)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        write_pickle_gz(instance, tmp.name)
        solver = LearningSolver(
            solver=internal_solver,
            simulate_perfect=True,
        )
        stats = solver.solve(PickleGzInstance(tmp.name))
        assert stats["Lower bound"] == stats["Objective: Predicted lower bound"]


def test_gap():
    assert LearningSolver._compute_gap(ub=0.0, lb=0.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=0.5) == 0.5
    assert LearningSolver._compute_gap(ub=1.0, lb=1.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=-1.0) is None
    assert LearningSolver._compute_gap(ub=1.0, lb=None) is None
    assert LearningSolver._compute_gap(ub=None, lb=1.0) is None
    assert LearningSolver._compute_gap(ub=None, lb=None) is None
