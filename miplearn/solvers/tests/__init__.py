#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any

from miplearn.features import Constraint
from miplearn.solvers.internal import InternalSolver


# NOTE:
# This file is in the main source folder, so that it can be called from Julia.


def run_internal_solver_tests(solver: InternalSolver) -> None:
    run_basic_usage_tests(solver.clone())
    run_warm_start_tests(solver.clone())
    run_infeasibility_tests(solver.clone())
    run_iteration_cb_tests(solver.clone())
    if solver.are_callbacks_supported():
        run_lazy_cb_tests(solver.clone())


def run_basic_usage_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    assert_equals(
        solver.get_variable_names(),
        ["x[0]", "x[1]", "x[2]", "x[3]"],
    )

    lp_stats = solver.solve_lp()
    assert not solver.is_infeasible()
    assert lp_stats["LP value"] is not None
    assert_equals(round(lp_stats["LP value"], 3), 1287.923)
    assert len(lp_stats["LP log"]) > 100

    solution = solver.get_solution()
    assert solution is not None
    assert solution["x[0]"] is not None
    assert solution["x[1]"] is not None
    assert solution["x[2]"] is not None
    assert solution["x[3]"] is not None
    assert_equals(round(solution["x[0]"], 3), 1.000)
    assert_equals(round(solution["x[1]"], 3), 0.923)
    assert_equals(round(solution["x[2]"], 3), 1.000)
    assert_equals(round(solution["x[3]"], 3), 0.000)

    mip_stats = solver.solve(
        tee=True,
        iteration_cb=None,
        lazy_cb=None,
        user_cut_cb=None,
    )
    assert not solver.is_infeasible()
    assert len(mip_stats["MIP log"]) > 100
    assert_equals(mip_stats["Lower bound"], 1183.0)
    assert_equals(mip_stats["Upper bound"], 1183.0)
    assert_equals(mip_stats["Sense"], "max")
    assert isinstance(mip_stats["Wallclock time"], float)

    solution = solver.get_solution()
    assert solution is not None
    assert solution["x[0]"] is not None
    assert solution["x[1]"] is not None
    assert solution["x[2]"] is not None
    assert solution["x[3]"] is not None
    assert_equals(solution["x[0]"], 1.0)
    assert_equals(solution["x[1]"], 0.0)
    assert_equals(solution["x[2]"], 1.0)
    assert_equals(solution["x[3]"], 1.0)

    assert_equals(
        solver.get_constraints(),
        {
            "eq_capacity": Constraint(
                lhs={"x[0]": 23.0, "x[1]": 26.0, "x[2]": 20.0, "x[3]": 18.0},
                rhs=67.0,
                sense="<",
            ),
        },
    )

    # Build a new constraint
    cut = Constraint(lhs={"x[0]": 1.0}, sense="<", rhs=0.0)
    assert not solver.is_constraint_satisfied(cut)

    # Add new constraint and verify that it is listed
    solver.add_constraint(cut, "cut")
    assert_equals(
        solver.get_constraints(),
        {
            "eq_capacity": Constraint(
                lhs={"x[0]": 23.0, "x[1]": 26.0, "x[2]": 20.0, "x[3]": 18.0},
                rhs=67.0,
                sense="<",
            ),
            "cut": Constraint(
                lhs={"x[0]": 1.0},
                rhs=0.0,
                sense="<",
            ),
        },
    )

    # New constraint should affect the solution
    stats = solver.solve()
    assert_equals(stats["Lower bound"], 1030.0)
    assert solver.is_constraint_satisfied(cut)

    # Verify slacks
    assert_equals(
        solver.get_inequality_slacks(),
        {"cut": 0.0, "eq_capacity": 3.0},
    )

    # Remove the new constraint
    solver.remove_constraint("cut")

    # New constraint should no longer affect solution
    stats = solver.solve()
    assert_equals(stats["Lower bound"], 1183.0)

    # Constraint should not be satisfied by current solution
    assert not solver.is_constraint_satisfied(cut)


def run_warm_start_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    solver.set_warm_start({"x[0]": 1.0, "x[1]": 0.0, "x[2]": 0.0, "x[3]": 1.0})
    stats = solver.solve(tee=True)
    if stats["Warm start value"] is not None:
        assert_equals(stats["Warm start value"], 725.0)

    solver.set_warm_start({"x[0]": 1.0, "x[1]": 1.0, "x[2]": 1.0, "x[3]": 1.0})
    stats = solver.solve(tee=True)
    assert stats["Warm start value"] is None

    solver.fix({"x[0]": 1.0, "x[1]": 0.0, "x[2]": 0.0, "x[3]": 1.0})
    stats = solver.solve(tee=True)
    assert_equals(stats["Lower bound"], 725.0)
    assert_equals(stats["Upper bound"], 725.0)


def run_infeasibility_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_infeasible()
    solver.set_instance(instance)
    mip_stats = solver.solve()
    assert solver.is_infeasible()
    assert solver.get_solution() is None
    assert mip_stats["Upper bound"] is None
    assert mip_stats["Lower bound"] is None
    lp_stats = solver.solve_lp()
    assert solver.get_solution() is None
    assert lp_stats["LP value"] is None


def run_iteration_cb_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_knapsack()
    solver.set_instance(instance)
    count = 0

    def custom_iteration_cb() -> bool:
        nonlocal count
        count += 1
        return count < 5

    solver.solve(iteration_cb=custom_iteration_cb)
    assert_equals(count, 5)


def run_lazy_cb_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()

    def lazy_cb(cb_solver: InternalSolver, cb_model: Any) -> None:
        relsol = cb_solver.get_solution()
        assert relsol is not None
        assert relsol["x[0]"] is not None
        if relsol["x[0]"] > 0:
            instance.enforce_lazy_constraint(cb_solver, cb_model, "cut")

    solver.set_instance(instance, model)
    solver.solve(lazy_cb=lazy_cb)
    solution = solver.get_solution()
    assert solution is not None
    assert_equals(solution["x[0]"], 0.0)


def assert_equals(left: Any, right: Any) -> None:
    assert left == right, f"{left} != {right}"
