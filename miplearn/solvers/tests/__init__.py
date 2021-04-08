#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn.solvers.internal import InternalSolver
from miplearn.instance.base import Instance
from typing import Any


def assert_equals(left: Any, right: Any) -> None:
    assert left == right, f"{left} != {right}"


def test_internal_solver(
    solver: InternalSolver,
    instance: Instance,
    model: Any,
) -> None:
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

    assert_equals(solver.get_constraint_ids(), ["eq_capacity"])
    assert_equals(
        solver.get_constraint_rhs("eq_capacity"),
        67.0,
    )
    assert_equals(
        solver.get_constraint_lhs("eq_capacity"),
        {
            "x[0]": 23.0,
            "x[1]": 26.0,
            "x[2]": 20.0,
            "x[3]": 18.0,
        },
    )
    assert_equals(solver.get_constraint_sense("eq_capacity"), "<")
