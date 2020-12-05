#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from io import StringIO

import pyomo.environ as pe

from miplearn import BasePyomoSolver, GurobiSolver
from miplearn.solvers import RedirectOutput
from . import _get_instance, _get_internal_solvers

logger = logging.getLogger(__name__)


def test_redirect_output():
    import sys

    original_stdout = sys.stdout
    io = StringIO()
    with RedirectOutput([io]):
        print("Hello world")
    assert sys.stdout == original_stdout
    assert io.getvalue() == "Hello world\n"


def test_internal_solver_warm_starts():
    for solver_class in _get_internal_solvers():
        logger.info("Solver: %s" % solver_class)
        instance = _get_instance(solver_class)
        model = instance.to_model()
        solver = solver_class()
        solver.set_instance(instance, model)
        solver.set_warm_start(
            {
                "x": {
                    0: 1.0,
                    1: 0.0,
                    2: 0.0,
                    3: 1.0,
                }
            }
        )
        stats = solver.solve(tee=True)
        assert stats["Warm start value"] == 725.0

        solver.set_warm_start(
            {
                "x": {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                }
            }
        )
        stats = solver.solve(tee=True)
        assert stats["Warm start value"] is None

        solver.fix(
            {
                "x": {
                    0: 1.0,
                    1: 0.0,
                    2: 0.0,
                    3: 1.0,
                }
            }
        )
        stats = solver.solve(tee=True)
        assert stats["Lower bound"] == 725.0
        assert stats["Upper bound"] == 725.0


def test_internal_solver():
    for solver_class in _get_internal_solvers():
        logger.info("Solver: %s" % solver_class)

        instance = _get_instance(solver_class)
        model = instance.to_model()
        solver = solver_class()
        solver.set_instance(instance, model)

        stats = solver.solve_lp()
        assert round(stats["Optimal value"], 3) == 1287.923

        solution = solver.get_solution()
        assert round(solution["x"][0], 3) == 1.000
        assert round(solution["x"][1], 3) == 0.923
        assert round(solution["x"][2], 3) == 1.000
        assert round(solution["x"][3], 3) == 0.000

        stats = solver.solve(tee=True)
        assert len(stats["Log"]) > 100
        assert stats["Lower bound"] == 1183.0
        assert stats["Upper bound"] == 1183.0
        assert stats["Sense"] == "max"
        assert isinstance(stats["Wallclock time"], float)
        assert isinstance(stats["Nodes"], int)

        solution = solver.get_solution()
        assert solution["x"][0] == 1.0
        assert solution["x"][1] == 0.0
        assert solution["x"][2] == 1.0
        assert solution["x"][3] == 1.0

        # Add a brand new constraint
        if isinstance(solver, BasePyomoSolver):
            model.cut = pe.Constraint(expr=model.x[0] <= 0.0, name="cut")
            solver.add_constraint(model.cut)
        elif isinstance(solver, GurobiSolver):
            x = model.getVarByName("x[0]")
            solver.add_constraint(x <= 0.0, name="cut")
        else:
            raise Exception("Illegal state")

        # New constraint should affect solution and should be listed in
        # constraint ids
        assert solver.get_constraint_ids() == ["eq_capacity", "cut"]
        stats = solver.solve()
        assert stats["Lower bound"] == 1030.0

        if isinstance(solver, GurobiSolver):
            # Extract new constraint
            cobj = solver.extract_constraint("cut")

            # New constraint should no longer affect solution and should no longer
            # be listed in constraint ids
            assert solver.get_constraint_ids() == ["eq_capacity"]
            stats = solver.solve()
            assert stats["Lower bound"] == 1183.0

            # New constraint should not be satisfied by current solution
            assert not solver.is_constraint_satisfied(cobj)

            # Re-add constraint
            solver.add_constraint(cobj)

            # Constraint should affect solution again
            assert solver.get_constraint_ids() == ["eq_capacity", "cut"]
            stats = solver.solve()
            assert stats["Lower bound"] == 1030.0

            # New constraint should now be satisfied
            assert solver.is_constraint_satisfied(cobj)


def test_iteration_cb():
    for solver_class in _get_internal_solvers():
        logger.info("Solver: %s" % solver_class)
        instance = _get_instance(solver_class)
        solver = solver_class()
        solver.set_instance(instance)
        count = 0

        def custom_iteration_cb():
            nonlocal count
            count += 1
            return count < 5

        solver.solve(iteration_cb=custom_iteration_cb)
        assert count == 5
