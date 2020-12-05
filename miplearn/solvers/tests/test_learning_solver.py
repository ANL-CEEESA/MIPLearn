#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import pickle
import tempfile
import os

from miplearn import DynamicLazyConstraintsComponent
from miplearn import LearningSolver

from . import _get_instance, _get_internal_solvers

logger = logging.getLogger(__name__)


def test_learning_solver():
    for mode in ["exact", "heuristic"]:
        for internal_solver in _get_internal_solvers():
            logger.info("Solver: %s" % internal_solver)
            instance = _get_instance(internal_solver)
            solver = LearningSolver(
                time_limit=300,
                gap_tolerance=1e-3,
                threads=1,
                solver=internal_solver,
                mode=mode,
            )

            solver.solve(instance)
            assert instance.solution["x"][0] == 1.0
            assert instance.solution["x"][1] == 0.0
            assert instance.solution["x"][2] == 1.0
            assert instance.solution["x"][3] == 1.0
            assert instance.lower_bound == 1183.0
            assert instance.upper_bound == 1183.0
            assert round(instance.lp_solution["x"][0], 3) == 1.000
            assert round(instance.lp_solution["x"][1], 3) == 0.923
            assert round(instance.lp_solution["x"][2], 3) == 1.000
            assert round(instance.lp_solution["x"][3], 3) == 0.000
            assert round(instance.lp_value, 3) == 1287.923
            assert instance.found_violated_lazy_constraints == []
            assert instance.found_violated_user_cuts == []
            assert len(instance.solver_log) > 100

            solver.fit([instance])
            solver.solve(instance)

            # Assert solver is picklable
            with tempfile.TemporaryFile() as file:
                pickle.dump(solver, file)


def test_parallel_solve():
    for internal_solver in _get_internal_solvers():
        instances = [_get_instance(internal_solver) for _ in range(10)]
        solver = LearningSolver(solver=internal_solver)
        results = solver.parallel_solve(instances, n_jobs=3)
        assert len(results) == 10
        for instance in instances:
            assert len(instance.solution["x"].keys()) == 4


def test_add_components():
    solver = LearningSolver(components=[])
    solver.add(DynamicLazyConstraintsComponent())
    solver.add(DynamicLazyConstraintsComponent())
    assert len(solver.components) == 1
    assert "DynamicLazyConstraintsComponent" in solver.components


def test_solve_fit_from_disk():
    for internal_solver in _get_internal_solvers():
        # Create instances and pickle them
        filenames = []
        for k in range(3):
            instance = _get_instance(internal_solver)
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
                filenames += [file.name]
                pickle.dump(instance, file)

        # Test: solve
        solver = LearningSolver(solver=internal_solver)
        solver.solve(filenames[0])
        with open(filenames[0], "rb") as file:
            instance = pickle.load(file)
            assert hasattr(instance, "solution")

        # Test: parallel_solve
        solver.parallel_solve(filenames)
        for filename in filenames:
            with open(filename, "rb") as file:
                instance = pickle.load(file)
                assert hasattr(instance, "solution")

        # Test: solve (with specified output)
        output = [f + ".out" for f in filenames]
        solver.solve(filenames[0], output=output[0])
        assert os.path.isfile(output[0])

        # Test: parallel_solve (with specified output)
        solver.parallel_solve(filenames, output=output)
        for filename in output:
            assert os.path.isfile(filename)

        # Delete temporary files
        for filename in filenames:
            os.remove(filename)
        for filename in output:
            os.remove(filename)
