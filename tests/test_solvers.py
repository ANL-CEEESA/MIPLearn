#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from tempfile import NamedTemporaryFile
from typing import Callable, Any

import numpy as np
import pytest

from miplearn.h5 import H5File
from miplearn.problems.setcover import (
    SetCoverData,
    build_setcover_model_gurobipy,
    build_setcover_model_pyomo,
)
from miplearn.solvers.abstract import AbstractModel

inf = float("inf")


@pytest.fixture
def data() -> SetCoverData:
    return SetCoverData(
        costs=np.array([5, 10, 12, 6, 8]),
        incidence_matrix=np.array(
            [
                [1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
            ],
        ),
    )


def test_gurobi(data: SetCoverData) -> None:
    _test_solver(build_setcover_model_gurobipy, data)


def test_pyomo_persistent(data: SetCoverData) -> None:
    _test_solver(lambda d: build_setcover_model_pyomo(d, "gurobi_persistent"), data)


def _test_solver(build_model: Callable, data: Any) -> None:
    _test_extract(build_model(data))
    _test_add_constr(build_model(data))
    _test_fix_vars(build_model(data))
    _test_infeasible(build_model(data))


def _test_extract(model: AbstractModel) -> None:
    with NamedTemporaryFile() as tempfile:
        with H5File(tempfile.name) as h5:

            def test_scalar(key: str, expected_value: Any) -> None:
                actual_value = h5.get_scalar(key)
                assert actual_value is not None
                assert actual_value == expected_value

            def test_array(key: str, expected_value: Any) -> None:
                actual_value = h5.get_array(key)
                assert actual_value is not None
                assert actual_value.tolist() == expected_value

            def test_sparse(key: str, expected_value: Any) -> None:
                actual_value = h5.get_sparse(key)
                assert actual_value is not None
                assert actual_value.todense().tolist() == expected_value

            model.extract_after_load(h5)
            test_sparse(
                "static_constr_lhs",
                [
                    [1.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                ],
            )
            test_array("static_constr_names", [b"eqs[0]", b"eqs[1]", b"eqs[2]"])
            test_array("static_constr_rhs", [1, 1, 1])
            test_array("static_constr_sense", [b">", b">", b">"])
            test_scalar("static_obj_offset", 0.0)
            test_scalar("static_sense", "min")
            test_array("static_var_lower_bounds", [0.0, 0.0, 0.0, 0.0, 0.0])
            test_array(
                "static_var_names",
                [
                    b"x[0]",
                    b"x[1]",
                    b"x[2]",
                    b"x[3]",
                    b"x[4]",
                ],
            )
            test_array("static_var_obj_coeffs", [5.0, 10.0, 12.0, 6.0, 8.0])
            test_array("static_var_types", [b"B", b"B", b"B", b"B", b"B"])
            test_array("static_var_upper_bounds", [1.0, 1.0, 1.0, 1.0, 1.0])

            relaxed = model.relax()
            relaxed.optimize()
            relaxed.extract_after_lp(h5)
            test_array("lp_constr_dual_values", [0, 5, 6])
            test_array("lp_constr_slacks", [1, 0, 0])
            test_scalar("lp_obj_value", 11.0)
            test_array("lp_var_reduced_costs", [0.0, 5.0, 6.0, 0.0, 2.0])
            test_array("lp_var_values", [1.0, 0.0, 0.0, 1.0, 0.0])
            if model._supports_basis_status:
                test_array("lp_var_basis_status", [b"B", b"L", b"L", b"B", b"L"])
                test_array("lp_constr_basis_status", [b"B", b"N", b"N"])
            if model._supports_sensitivity_analysis:
                test_array("lp_constr_sa_rhs_up", [2, 1, 1])
                test_array("lp_constr_sa_rhs_down", [-inf, 0, 0])
                test_array("lp_var_sa_obj_up", [10.0, inf, inf, 8.0, inf])
                test_array("lp_var_sa_obj_down", [0.0, 5.0, 6.0, 0.0, 6.0])
                test_array("lp_var_sa_ub_up", [inf, inf, inf, inf, inf])
                test_array("lp_var_sa_ub_down", [1.0, 0.0, 0.0, 1.0, 0.0])
                test_array("lp_var_sa_lb_up", [1.0, 1.0, 1.0, 1.0, 1.0])
                test_array("lp_var_sa_lb_down", [-inf, 0.0, 0.0, -inf, 0.0])
            lp_wallclock_time = h5.get_scalar("lp_wallclock_time")
            assert lp_wallclock_time is not None
            assert lp_wallclock_time >= 0

            model.optimize()
            model.extract_after_mip(h5)
            test_array("mip_constr_slacks", [1, 0, 0])
            test_array("mip_var_values", [1.0, 0.0, 0.0, 1.0, 0.0])
            test_scalar("mip_gap", 0)
            test_scalar("mip_obj_bound", 11.0)
            test_scalar("mip_obj_value", 11.0)
            mip_wallclock_time = h5.get_scalar("mip_wallclock_time")
            assert mip_wallclock_time is not None
            assert mip_wallclock_time > 0
            if model._supports_node_count:
                count = h5.get_scalar("mip_node_count")
                assert count is not None
                assert count >= 0
            if model._supports_solution_pool:
                pool_var_values = h5.get_array("pool_var_values")
                pool_obj_values = h5.get_array("pool_obj_values")
                assert pool_var_values is not None
                assert pool_obj_values is not None
                assert len(pool_obj_values.shape) == 1
                n_sols = len(pool_obj_values)
                assert pool_var_values.shape == (n_sols, 5)


def _test_add_constr(model: AbstractModel) -> None:
    with NamedTemporaryFile() as tempfile:
        with H5File(tempfile.name) as h5:
            model.add_constrs(
                np.array([b"x[2]", b"x[3]"], dtype="S"),
                np.array([[0, 1], [1, 0]]),
                np.array(["=", "="], dtype="S"),
                np.array([0, 0]),
            )
            model.optimize()
            model.extract_after_mip(h5)
            mip_var_values = h5.get_array("mip_var_values")
            assert mip_var_values is not None
            assert mip_var_values.tolist() == [1, 0, 0, 0, 1]


def _test_fix_vars(model: AbstractModel) -> None:
    with NamedTemporaryFile() as tempfile:
        with H5File(tempfile.name) as h5:
            model.fix_variables(
                var_names=np.array([b"x[2]", b"x[3]"], dtype="S"),
                var_values=np.array([0, 0]),
            )
            model.optimize()
            model.extract_after_mip(h5)
            mip_var_values = h5.get_array("mip_var_values")
            assert mip_var_values is not None
            assert mip_var_values.tolist() == [1, 0, 0, 0, 1]


def _test_infeasible(model: AbstractModel) -> None:
    with NamedTemporaryFile() as tempfile:
        with H5File(tempfile.name) as h5:
            model.fix_variables(
                var_names=np.array([b"x[0]", b"x[3]"], dtype="S"),
                var_values=np.array([0, 0]),
            )
            model.optimize()
            model.extract_after_mip(h5)
            assert h5.get_array("mip_var_values") is None
