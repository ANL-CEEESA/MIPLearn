#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import pickle
import tempfile

from miplearn import BranchPriorityComponent
from miplearn import LearningSolver

from . import _get_instance, _get_internal_solvers

logger = logging.getLogger(__name__)


def test_learning_solver():
    for mode in ["exact", "heuristic"]:
        for internal_solver in _get_internal_solvers():
            logger.info("Solver: %s" % internal_solver)
            instance = _get_instance(internal_solver)
            solver = LearningSolver(time_limit=300,
                                    gap_tolerance=1e-3,
                                    threads=1,
                                    solver=internal_solver,
                                    mode=mode)

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
    solver.add(BranchPriorityComponent())
    solver.add(BranchPriorityComponent())
    assert len(solver.components) == 1
    assert "BranchPriorityComponent" in solver.components
