#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import pickle
import tempfile
import os

from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.learning import LearningSolver
from . import _get_knapsack_instance, _get_internal_solvers

logger = logging.getLogger(__name__)


def test_learning_solver():
    for mode in ["exact", "heuristic"]:
        for internal_solver in _get_internal_solvers():
            logger.info("Solver: %s" % internal_solver)
            instance = _get_knapsack_instance(internal_solver)
            solver = LearningSolver(
                solver=internal_solver,
                mode=mode,
            )

            solver.solve(instance)
            data = instance.training_data[0]
            assert data["Solution"]["x"][0] == 1.0
            assert data["Solution"]["x"][1] == 0.0
            assert data["Solution"]["x"][2] == 1.0
            assert data["Solution"]["x"][3] == 1.0
            assert data["Lower bound"] == 1183.0
            assert data["Upper bound"] == 1183.0
            assert round(data["LP solution"]["x"][0], 3) == 1.000
            assert round(data["LP solution"]["x"][1], 3) == 0.923
            assert round(data["LP solution"]["x"][2], 3) == 1.000
            assert round(data["LP solution"]["x"][3], 3) == 0.000
            assert round(data["LP value"], 3) == 1287.923
            assert len(data["MIP log"]) > 100

            solver.fit([instance])
            solver.solve(instance)

            # Assert solver is picklable
            with tempfile.TemporaryFile() as file:
                pickle.dump(solver, file)


def test_solve_without_lp():
    for internal_solver in _get_internal_solvers():
        logger.info("Solver: %s" % internal_solver)
        instance = _get_knapsack_instance(internal_solver)
        solver = LearningSolver(
            solver=internal_solver,
            solve_lp_first=False,
        )
        solver.solve(instance)
        solver.fit([instance])
        solver.solve(instance)


def test_parallel_solve():
    for internal_solver in _get_internal_solvers():
        instances = [_get_knapsack_instance(internal_solver) for _ in range(10)]
        solver = LearningSolver(solver=internal_solver)
        results = solver.parallel_solve(instances, n_jobs=3)
        assert len(results) == 10
        for instance in instances:
            data = instance.training_data[0]
            assert len(data["Solution"]["x"].keys()) == 4


def test_solve_fit_from_disk():
    for internal_solver in _get_internal_solvers():
        # Create instances and pickle them
        filenames = []
        for k in range(3):
            instance = _get_knapsack_instance(internal_solver)
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
                filenames += [file.name]
                pickle.dump(instance, file)

        # Test: solve
        solver = LearningSolver(solver=internal_solver)
        solver.solve(filenames[0])
        with open(filenames[0], "rb") as file:
            instance = pickle.load(file)
            assert len(instance.training_data) > 0

        # Test: parallel_solve
        solver.parallel_solve(filenames)
        for filename in filenames:
            with open(filename, "rb") as file:
                instance = pickle.load(file)
                assert len(instance.training_data) > 0

        # Test: solve (with specified output)
        output = [f + ".out" for f in filenames]
        solver.solve(
            filenames[0],
            output_filename=output[0],
        )
        assert os.path.isfile(output[0])

        # Test: parallel_solve (with specified output)
        solver.parallel_solve(
            filenames,
            output_filenames=output,
        )
        for filename in output:
            assert os.path.isfile(filename)

        # Delete temporary files
        for filename in filenames:
            os.remove(filename)
        for filename in output:
            os.remove(filename)


def test_simulate_perfect():
    internal_solver = GurobiSolver
    instance = _get_knapsack_instance(internal_solver)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(instance, tmp)
        tmp.flush()
        solver = LearningSolver(
            solver=internal_solver,
            simulate_perfect=True,
        )
        stats = solver.solve(tmp.name)
        assert stats["Lower bound"] == stats["Predicted LB"]


def test_gap():
    assert LearningSolver._compute_gap(ub=0.0, lb=0.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=0.5) == 0.5
    assert LearningSolver._compute_gap(ub=1.0, lb=1.0) == 0.0
    assert LearningSolver._compute_gap(ub=1.0, lb=-1.0) is None
    assert LearningSolver._compute_gap(ub=1.0, lb=None) is None
    assert LearningSolver._compute_gap(ub=None, lb=1.0) is None
    assert LearningSolver._compute_gap(ub=None, lb=None) is None
