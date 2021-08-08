#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any, List

import numpy as np

from miplearn.solvers.internal import InternalSolver, Variables, Constraints

inf = float("inf")


# NOTE:
# This file is in the main source folder, so that it can be called from Julia.


def _filter_attrs(allowed_keys: List[str], obj: Any) -> Any:
    for key in obj.__dict__.keys():
        if key not in allowed_keys:
            setattr(obj, key, None)
    return obj


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
        solver.get_variables(),
        Variables(
            names=np.array(["x[0]", "x[1]", "x[2]", "x[3]", "z"], dtype="S"),
            lower_bounds=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0, 1.0, 1.0, 67.0]),
            types=np.array(["B", "B", "B", "B", "C"], dtype="S"),
            obj_coeffs=np.array([505.0, 352.0, 458.0, 220.0, 0.0]),
        ),
    )

    # Fetch constraints (after-load)
    assert_equals(
        solver.get_constraints(),
        Constraints(
            names=["eq_capacity"],
            rhs=np.array([0.0]),
            lhs=[
                [
                    (b"x[0]", 23.0),
                    (b"x[1]", 26.0),
                    (b"x[2]", 20.0),
                    (b"x[3]", 18.0),
                    (b"z", -1.0),
                ],
            ],
            senses=["="],
        ),
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

    # Fetch variables (after-lp)
    assert_equals(
        solver.get_variables(with_static=False),
        _filter_attrs(
            solver.get_variable_attrs(),
            Variables(
                names=np.array(["x[0]", "x[1]", "x[2]", "x[3]", "z"], dtype="S"),
                basis_status=np.array(["U", "B", "U", "L", "U"], dtype="S"),
                reduced_costs=np.array(
                    [193.615385, 0.0, 187.230769, -23.692308, 13.538462]
                ),
                sa_lb_down=np.array([-inf, -inf, -inf, -0.111111, -inf]),
                sa_lb_up=np.array([1.0, 0.923077, 1.0, 1.0, 67.0]),
                sa_obj_down=np.array(
                    [311.384615, 317.777778, 270.769231, -inf, -13.538462]
                ),
                sa_obj_up=np.array([inf, 570.869565, inf, 243.692308, inf]),
                sa_ub_down=np.array([0.913043, 0.923077, 0.9, 0.0, 43.0]),
                sa_ub_up=np.array([2.043478, inf, 2.2, inf, 69.0]),
                values=np.array([1.0, 0.923077, 1.0, 0.0, 67.0]),
            ),
        ),
    )

    # Fetch constraints (after-lp)
    assert_equals(
        solver.get_constraints(with_static=False),
        _filter_attrs(
            solver.get_constraint_attrs(),
            Constraints(
                basis_status=["N"],
                dual_values=np.array([13.538462]),
                names=["eq_capacity"],
                sa_rhs_down=np.array([-24.0]),
                sa_rhs_up=np.array([2.0]),
                slacks=np.array([0.0]),
            ),
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
        solver.get_variables(with_static=False),
        _filter_attrs(
            solver.get_variable_attrs(),
            Variables(
                names=np.array(["x[0]", "x[1]", "x[2]", "x[3]", "z"], dtype="S"),
                values=np.array([1.0, 0.0, 1.0, 1.0, 61.0]),
            ),
        ),
    )

    # Fetch constraints (after-mip)
    assert_equals(
        solver.get_constraints(with_static=False),
        _filter_attrs(
            solver.get_constraint_attrs(),
            Constraints(
                names=["eq_capacity"],
                slacks=np.array([0.0]),
            ),
        ),
    )

    # Build new constraint and verify that it is violated
    cf = Constraints(
        names=["cut"],
        lhs=[[(b"x[0]", 1.0)]],
        rhs=np.array([0.0]),
        senses=["<"],
    )
    assert_equals(solver.are_constraints_satisfied(cf), [False])

    # Add constraint and verify it affects solution
    solver.add_constraints(cf)
    assert_equals(
        solver.get_constraints(with_static=True),
        _filter_attrs(
            solver.get_constraint_attrs(),
            Constraints(
                names=["eq_capacity", "cut"],
                rhs=np.array([0.0, 0.0]),
                lhs=[
                    [
                        (b"x[0]", 23.0),
                        (b"x[1]", 26.0),
                        (b"x[2]", 20.0),
                        (b"x[3]", 18.0),
                        (b"z", -1.0),
                    ],
                    [
                        (b"x[0]", 1.0),
                    ],
                ],
                senses=["=", "<"],
            ),
        ),
    )
    stats = solver.solve()
    assert_equals(stats.mip_lower_bound, 1030.0)
    assert_equals(solver.are_constraints_satisfied(cf), [True])

    # Remove the new constraint
    solver.remove_constraints(["cut"])

    # New constraint should no longer affect solution
    stats = solver.solve()
    assert_equals(stats.mip_lower_bound, 1183.0)


def run_warm_start_tests(solver: InternalSolver) -> None:
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    solver.set_warm_start({b"x[0]": 1.0, b"x[1]": 0.0, b"x[2]": 0.0, b"x[3]": 1.0})
    stats = solver.solve(tee=True)
    if stats.mip_warm_start_value is not None:
        assert_equals(stats.mip_warm_start_value, 725.0)

    solver.set_warm_start({b"x[0]": 1.0, b"x[1]": 1.0, b"x[2]": 1.0, b"x[3]": 1.0})
    stats = solver.solve(tee=True)
    assert stats.mip_warm_start_value is None

    solver.fix({b"x[0]": 1.0, b"x[1]": 0.0, b"x[2]": 0.0, b"x[3]": 1.0})
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
        assert relsol[b"x[0]"] is not None
        if relsol[b"x[0]"] > 0:
            instance.enforce_lazy_constraint(cb_solver, cb_model, "cut")

    solver.set_instance(instance, model)
    solver.solve(lazy_cb=lazy_cb)
    solution = solver.get_solution()
    assert solution is not None
    assert_equals(solution[b"x[0]"], 0.0)


def _equals_preprocess(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype == "float64":
            return np.round(obj, decimals=6).tolist()
        else:
            return obj.tolist()
    elif isinstance(obj, (int, str, bool, np.bool_, np.bytes_, bytes)):
        return obj
    elif isinstance(obj, float):
        return round(obj, 6)
    elif isinstance(obj, list):
        return [_equals_preprocess(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_equals_preprocess(i) for i in obj)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: _equals_preprocess(v) for (k, v) in obj.items()}
    else:
        for key in obj.__dict__.keys():
            obj.__dict__[key] = _equals_preprocess(obj.__dict__[key])
        return obj


def assert_equals(left: Any, right: Any) -> None:
    left = _equals_preprocess(left)
    right = _equals_preprocess(right)
    assert left == right, f"left:\n{left}\nright:\n{right}"
