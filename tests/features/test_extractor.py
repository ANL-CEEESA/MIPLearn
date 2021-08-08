#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import sys
import time
from typing import Any

import numpy as np
import gurobipy as gp

from miplearn.features.extractor import FeaturesExtractor
from miplearn.features.sample import MemorySample, Hdf5Sample
from miplearn.instance.base import Instance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.internal import Variables, Constraints
from miplearn.solvers.tests import assert_equals
import cProfile

inf = float("inf")


def test_knapsack() -> None:
    solver = GurobiSolver()
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    extractor = FeaturesExtractor()
    sample = MemorySample()

    # after-load
    # -------------------------------------------------------
    extractor.extract_after_load_features(instance, solver, sample)
    assert_equals(
        sample.get_vector("static_var_names"),
        np.array(["x[0]", "x[1]", "x[2]", "x[3]", "z"], dtype="S"),
    )
    assert_equals(
        sample.get_vector("static_var_lower_bounds"), [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert_equals(
        sample.get_vector("static_var_obj_coeffs"), [505.0, 352.0, 458.0, 220.0, 0.0]
    )
    assert_equals(
        sample.get_array("static_var_types"),
        np.array(["B", "B", "B", "B", "C"], dtype="S"),
    )
    assert_equals(
        sample.get_vector("static_var_upper_bounds"), [1.0, 1.0, 1.0, 1.0, 67.0]
    )
    assert_equals(
        sample.get_vector("static_var_categories"),
        ["default", "default", "default", "default", None],
    )
    assert sample.get_vector_list("static_var_features") is not None
    assert_equals(sample.get_vector("static_constr_names"), ["eq_capacity"])
    # assert_equals(
    #     sample.get_vector("static_constr_lhs"),
    #     [
    #         [
    #             ("x[0]", 23.0),
    #             ("x[1]", 26.0),
    #             ("x[2]", 20.0),
    #             ("x[3]", 18.0),
    #             ("z", -1.0),
    #         ],
    #     ],
    # )
    assert_equals(sample.get_vector("static_constr_rhs"), [0.0])
    assert_equals(sample.get_vector("static_constr_senses"), ["="])
    assert_equals(sample.get_vector("static_constr_features"), [None])
    assert_equals(sample.get_vector("static_constr_categories"), ["eq_capacity"])
    assert_equals(sample.get_vector("static_constr_lazy"), [False])
    assert_equals(sample.get_vector("static_instance_features"), [67.0, 21.75])
    assert_equals(sample.get_scalar("static_constr_lazy_count"), 0)

    # after-lp
    # -------------------------------------------------------
    solver.solve_lp()
    extractor.extract_after_lp_features(solver, sample)
    assert_equals(
        sample.get_array("lp_var_basis_status"),
        np.array(["U", "B", "U", "L", "U"], dtype="S"),
    )
    assert_equals(
        sample.get_vector("lp_var_reduced_costs"),
        [193.615385, 0.0, 187.230769, -23.692308, 13.538462],
    )
    assert_equals(
        sample.get_vector("lp_var_sa_lb_down"),
        [-inf, -inf, -inf, -0.111111, -inf],
    )
    assert_equals(
        sample.get_vector("lp_var_sa_lb_up"),
        [1.0, 0.923077, 1.0, 1.0, 67.0],
    )
    assert_equals(
        sample.get_vector("lp_var_sa_obj_down"),
        [311.384615, 317.777778, 270.769231, -inf, -13.538462],
    )
    assert_equals(
        sample.get_vector("lp_var_sa_obj_up"),
        [inf, 570.869565, inf, 243.692308, inf],
    )
    assert_equals(
        sample.get_vector("lp_var_sa_ub_down"), [0.913043, 0.923077, 0.9, 0.0, 43.0]
    )
    assert_equals(sample.get_vector("lp_var_sa_ub_up"), [2.043478, inf, 2.2, inf, 69.0])
    assert_equals(sample.get_vector("lp_var_values"), [1.0, 0.923077, 1.0, 0.0, 67.0])
    assert sample.get_vector_list("lp_var_features") is not None
    assert_equals(sample.get_vector("lp_constr_basis_status"), ["N"])
    assert_equals(sample.get_vector("lp_constr_dual_values"), [13.538462])
    assert_equals(sample.get_vector("lp_constr_sa_rhs_down"), [-24.0])
    assert_equals(sample.get_vector("lp_constr_sa_rhs_up"), [2.0])
    assert_equals(sample.get_vector("lp_constr_slacks"), [0.0])

    # after-mip
    # -------------------------------------------------------
    solver.solve()
    extractor.extract_after_mip_features(solver, sample)
    assert_equals(sample.get_vector("mip_var_values"), [1.0, 0.0, 1.0, 1.0, 61.0])
    assert_equals(sample.get_vector("mip_constr_slacks"), [0.0])


def test_constraint_getindex() -> None:
    cf = Constraints(
        names=["c1", "c2", "c3"],
        rhs=np.array([1.0, 2.0, 3.0]),
        senses=["=", "<", ">"],
        lhs=[
            [
                (b"x1", 1.0),
                (b"x2", 1.0),
            ],
            [
                (b"x2", 2.0),
                (b"x3", 2.0),
            ],
            [
                (b"x3", 3.0),
                (b"x4", 3.0),
            ],
        ],
    )
    assert_equals(
        cf[[True, False, True]],
        Constraints(
            names=["c1", "c3"],
            rhs=np.array([1.0, 3.0]),
            senses=["=", ">"],
            lhs=[
                [
                    (b"x1", 1.0),
                    (b"x2", 1.0),
                ],
                [
                    (b"x3", 3.0),
                    (b"x4", 3.0),
                ],
            ],
        ),
    )


def test_assert_equals() -> None:
    assert_equals("hello", "hello")
    assert_equals([1.0, 2.0], [1.0, 2.0])
    assert_equals(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert_equals(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert_equals(
        Variables(values=np.array([1.0, 2.0])),  # type: ignore
        Variables(values=np.array([1.0, 2.0])),  # type: ignore
    )
    assert_equals(np.array([True, True]), [True, True])
    assert_equals((1.0,), (1.0,))
    assert_equals({"x": 10}, {"x": 10})


class MpsInstance(Instance):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename

    def to_model(self) -> Any:
        return gp.read(self.filename)


if __name__ == "__main__":
    solver = GurobiSolver()
    instance = MpsInstance(sys.argv[1])
    solver.set_instance(instance)
    solver.solve_lp(tee=True)
    extractor = FeaturesExtractor(with_lhs=False)
    sample = Hdf5Sample("tmp/prof.h5", mode="w")

    def run() -> None:
        extractor.extract_after_load_features(instance, solver, sample)
        extractor.extract_after_lp_features(solver, sample)

    cProfile.run("run()", filename="tmp/prof")
