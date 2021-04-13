#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any, Dict
import numpy as np

from miplearn.features import Constraint, Variable
from miplearn.solvers.internal import InternalSolver

inf = float("inf")

# NOTE:
# This file is in the main source folder, so that it can be called from Julia.


def _round_constraints(constraints: Dict[str, Constraint]) -> Dict[str, Constraint]:
    for (cname, c) in constraints.items():
        for attr in ["slack", "dual_value"]:
            if getattr(c, attr) is not None:
                setattr(c, attr, round(getattr(c, attr), 6))
    return constraints


def _round_variables(vars: Dict[str, Variable]) -> Dict[str, Variable]:
    for (cname, c) in vars.items():
        for attr in [
            "upper_bound",
            "lower_bound",
            "obj_coeff",
            "value",
            "reduced_cost",
            "sa_obj_up",
            "sa_obj_down",
            "sa_ub_up",
            "sa_ub_down",
            "sa_lb_up",
            "sa_lb_down",
        ]:
            if getattr(c, attr) is not None:
                setattr(c, attr, round(getattr(c, attr), 6))
        if c.alvarez_2017 is not None:
            c.alvarez_2017 = list(np.round(c.alvarez_2017, 6))
    return vars


def _remove_unsupported_constr_attrs(
    solver: InternalSolver,
    constraints: Dict[str, Constraint],
) -> Dict[str, Constraint]:
    for (cname, c) in constraints.items():
        to_remove = []
        for k in c.__dict__.keys():
            if k not in solver.get_constraint_attrs():
                to_remove.append(k)
        for k in to_remove:
            setattr(c, k, None)
    return constraints


def _remove_unsupported_var_attrs(
    solver: InternalSolver,
    variables: Dict[str, Variable],
) -> Dict[str, Variable]:
    for (cname, c) in variables.items():
        to_remove = []
        for k in c.__dict__.keys():
            if k not in solver.get_variable_attrs():
                to_remove.append(k)
        for k in to_remove:
            setattr(c, k, None)
    return variables


def run_internal_solver_tests(solver: InternalSolver) -> None:
    run_basic_usage_tests(solver.clone())
    run_warm_start_tests(solver.clone())
    run_infeasibility_tests(solver.clone())
    run_iteration_cb_tests(solver.clone())
    if solver.are_callbacks_supported():
        run_lazy_cb_tests(solver.clone())


def run_basic_usage_tests(solver: InternalSolver) -> None:
    # Create and set instance
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)

    # Fetch variables (after-load)
    assert_equals(
        _round_variables(solver.get_variables()),
        _remove_unsupported_var_attrs(
            solver,
            {
                "x[0]": Variable(
                    lower_bound=0.0,
                    obj_coeff=505.0,
                    type="B",
                    upper_bound=1.0,
                ),
                "x[1]": Variable(
                    lower_bound=0.0,
                    obj_coeff=352.0,
                    type="B",
                    upper_bound=1.0,
                ),
                "x[2]": Variable(
                    lower_bound=0.0,
                    obj_coeff=458.0,
                    type="B",
                    upper_bound=1.0,
                ),
                "x[3]": Variable(
                    lower_bound=0.0,
                    obj_coeff=220.0,
                    type="B",
                    upper_bound=1.0,
                ),
                "z": Variable(
                    lower_bound=0.0,
                    obj_coeff=0.0,
                    type="C",
                    upper_bound=67.0,
                ),
            },
        ),
    )

    # Fetch constraints (after-load)
    assert_equals(
        _round_constraints(solver.get_constraints()),
        {
            "eq_capacity": Constraint(
                lazy=False,
                lhs={"x[0]": 23.0, "x[1]": 26.0, "x[2]": 20.0, "x[3]": 18.0, "z": -1.0},
                rhs=0.0,
                sense="=",
            )
        },
    )

    # Solve linear programming relaxation
    lp_stats = solver.solve_lp()
    assert not solver.is_infeasible()
    assert lp_stats.lp_value is not None
    assert_equals(round(lp_stats.lp_value, 3), 1287.923)
    assert lp_stats.lp_log is not None
    assert len(lp_stats.lp_log) > 100
    assert lp_stats.lp_wallclock_time is not None
    assert lp_stats.lp_wallclock_time > 0

    # Fetch variables (after-load)
    assert_equals(
        _round_variables(solver.get_variables()),
        _remove_unsupported_var_attrs(
            solver,
            {
                "x[0]": Variable(
                    basis_status="U",
                    lower_bound=0.0,
                    obj_coeff=505.0,
                    reduced_cost=193.615385,
                    sa_lb_down=-inf,
                    sa_lb_up=1.0,
                    sa_obj_down=311.384615,
                    sa_obj_up=inf,
                    sa_ub_down=0.913043,
                    sa_ub_up=2.043478,
                    type="B",
                    upper_bound=1.0,
                    value=1.0,
                ),
                "x[1]": Variable(
                    basis_status="B",
                    lower_bound=0.0,
                    obj_coeff=352.0,
                    reduced_cost=0.0,
                    sa_lb_down=-inf,
                    sa_lb_up=0.923077,
                    sa_obj_down=317.777778,
                    sa_obj_up=570.869565,
                    sa_ub_down=0.923077,
                    sa_ub_up=inf,
                    type="B",
                    upper_bound=1.0,
                    value=0.923077,
                ),
                "x[2]": Variable(
                    basis_status="U",
                    lower_bound=0.0,
                    obj_coeff=458.0,
                    reduced_cost=187.230769,
                    sa_lb_down=-inf,
                    sa_lb_up=1.0,
                    sa_obj_down=270.769231,
                    sa_obj_up=inf,
                    sa_ub_down=0.9,
                    sa_ub_up=2.2,
                    type="B",
                    upper_bound=1.0,
                    value=1.0,
                ),
                "x[3]": Variable(
                    basis_status="L",
                    lower_bound=0.0,
                    obj_coeff=220.0,
                    reduced_cost=-23.692308,
                    sa_lb_down=-0.111111,
                    sa_lb_up=1.0,
                    sa_obj_down=-inf,
                    sa_obj_up=243.692308,
                    sa_ub_down=0.0,
                    sa_ub_up=inf,
                    type="B",
                    upper_bound=1.0,
                    value=0.0,
                ),
                "z": Variable(
                    basis_status="U",
                    lower_bound=0.0,
                    obj_coeff=0.0,
                    reduced_cost=13.538462,
                    sa_lb_down=-inf,
                    sa_lb_up=67.0,
                    sa_obj_down=-13.538462,
                    sa_obj_up=inf,
                    sa_ub_down=43.0,
                    sa_ub_up=69.0,
                    type="C",
                    upper_bound=67.0,
                    value=67.0,
                ),
            },
        ),
    )

    # Fetch constraints (after-lp)
    assert_equals(
        _round_constraints(solver.get_constraints()),
        _remove_unsupported_constr_attrs(
            solver,
            {
                "eq_capacity": Constraint(
                    basis_status="N",
                    dual_value=13.538462,
                    lazy=False,
                    lhs={
                        "x[0]": 23.0,
                        "x[1]": 26.0,
                        "x[2]": 20.0,
                        "x[3]": 18.0,
                        "z": -1.0,
                    },
                    rhs=0.0,
                    sa_rhs_down=-24.0,
                    sa_rhs_up=1.9999999999999987,
                    sense="=",
                    slack=0.0,
                )
            },
        ),
    )

    # Solve MIP
    mip_stats = solver.solve(
        tee=True,
    )
    assert not solver.is_infeasible()
    assert mip_stats.mip_log is not None
    assert len(mip_stats.mip_log) > 100
    assert mip_stats.mip_lower_bound is not None
    assert_equals(mip_stats.mip_lower_bound, 1183.0)
    assert mip_stats.mip_upper_bound is not None
    assert_equals(mip_stats.mip_upper_bound, 1183.0)
    assert mip_stats.mip_sense is not None
    assert_equals(mip_stats.mip_sense, "max")
    assert mip_stats.mip_wallclock_time is not None
    assert isinstance(mip_stats.mip_wallclock_time, float)
    assert mip_stats.mip_wallclock_time > 0

    # Fetch variables (after-mip)
    assert_equals(
        _round_variables(solver.get_variables(with_static=False)),
        _remove_unsupported_var_attrs(
            solver,
            {
                "x[0]": Variable(value=1.0),
                "x[1]": Variable(value=0.0),
                "x[2]": Variable(value=1.0),
                "x[3]": Variable(value=1.0),
                "z": Variable(value=61.0),
            },
        ),
    )

    # Fetch constraints (after-mip)
    assert_equals(
        _round_constraints(solver.get_constraints(with_static=False)),
        {"eq_capacity": Constraint(slack=0.0)},
    )

    # Build a new constraint
    cut = Constraint(lhs={"x[0]": 1.0}, sense="<", rhs=0.0)
    assert not solver.is_constraint_satisfied(cut)

    # Add new constraint and verify that it is listed. Modifying the model should
    # also clear the current solution.
    solver.add_constraint(cut, "cut")
    assert_equals(
        _round_constraints(solver.get_constraints()),
        {
            "eq_capacity": Constraint(
                lazy=False,
                lhs={"x[0]": 23.0, "x[1]": 26.0, "x[2]": 20.0, "x[3]": 18.0, "z": -1.0},
                rhs=0.0,
                sense="=",
            ),
            "cut": Constraint(
                lazy=False,
                lhs={"x[0]": 1.0},
                rhs=0.0,
                sense="<",
            ),
        },
    )

    # Re-solve MIP and verify that constraint affects the solution
    stats = solver.solve()
    assert_equals(stats.mip_lower_bound, 1030.0)
    assert solver.is_constraint_satisfied(cut)

    # Remove the new constraint
    solver.remove_constraint("cut")

    # New constraint should no longer affect solution
    stats = solver.solve()
    assert_equals(stats.mip_lower_bound, 1183.0)


def run_warm_start_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    solver.set_warm_start({"x[0]": 1.0, "x[1]": 0.0, "x[2]": 0.0, "x[3]": 1.0})
    stats = solver.solve(tee=True)
    if stats.mip_warm_start_value is not None:
        assert_equals(stats.mip_warm_start_value, 725.0)

    solver.set_warm_start({"x[0]": 1.0, "x[1]": 1.0, "x[2]": 1.0, "x[3]": 1.0})
    stats = solver.solve(tee=True)
    assert stats.mip_warm_start_value is None

    solver.fix({"x[0]": 1.0, "x[1]": 0.0, "x[2]": 0.0, "x[3]": 1.0})
    stats = solver.solve(tee=True)
    assert_equals(stats.mip_lower_bound, 725.0)
    assert_equals(stats.mip_upper_bound, 725.0)


def run_infeasibility_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_infeasible()
    solver.set_instance(instance)
    mip_stats = solver.solve()
    assert solver.is_infeasible()
    assert solver.get_solution() is None
    assert mip_stats.mip_upper_bound is None
    assert mip_stats.mip_lower_bound is None
    lp_stats = solver.solve_lp()
    assert solver.get_solution() is None
    assert lp_stats.lp_value is None


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
    assert left == right, f"left:\n{left}\nright:\n{right}"
