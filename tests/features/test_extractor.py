#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import os
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
        sample.get_array("static_instance_features"),
        np.array([67.0, 21.75]),
    )
    assert_equals(
        sample.get_array("static_var_names"),
        np.array(["x[0]", "x[1]", "x[2]", "x[3]", "z"], dtype="S"),
    )
    assert_equals(
        sample.get_array("static_var_lower_bounds"),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert_equals(
        sample.get_array("static_var_obj_coeffs"),
        np.array([505.0, 352.0, 458.0, 220.0, 0.0]),
    )
    assert_equals(
        sample.get_array("static_var_types"),
        np.array(["B", "B", "B", "B", "C"], dtype="S"),
    )
    assert_equals(
        sample.get_array("static_var_upper_bounds"),
        np.array([1.0, 1.0, 1.0, 1.0, 67.0]),
    )
    assert_equals(
        sample.get_array("static_var_categories"),
        np.array(["default", "default", "default", "default", ""], dtype="S"),
    )
    assert_equals(
        sample.get_array("static_var_features"),
        np.array(
            [
                [23.0, 505.0, 1.0, 0.32899, 0.0, 505.0, 1.0],
                [26.0, 352.0, 1.0, 0.229316, 0.0, 352.0, 1.0],
                [20.0, 458.0, 1.0, 0.298371, 0.0, 458.0, 1.0],
                [18.0, 220.0, 1.0, 0.143322, 0.0, 220.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 67.0],
            ]
        ),
    )
    assert_equals(
        sample.get_array("static_constr_names"),
        np.array(["eq_capacity"], dtype="S"),
    )
    # assert_equals(
    #     sample.get_array("static_constr_lhs"),
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
    assert_equals(
        sample.get_array("static_constr_rhs"),
        np.array([0.0]),
    )
    assert_equals(
        sample.get_array("static_constr_senses"),
        np.array(["="], dtype="S"),
    )
    assert_equals(
        sample.get_array("static_constr_features"),
        np.array([[0.0]]),
    )
    assert_equals(
        sample.get_array("static_constr_categories"),
        np.array(["eq_capacity"], dtype="S"),
    )
    assert_equals(
        sample.get_array("static_constr_lazy"),
        np.array([False]),
    )
    assert_equals(
        sample.get_array("static_instance_features"),
        np.array([67.0, 21.75]),
    )
    assert_equals(sample.get_scalar("static_constr_lazy_count"), 0)

    # after-lp
    # -------------------------------------------------------
    lp_stats = solver.solve_lp()
    extractor.extract_after_lp_features(solver, sample, lp_stats)
    assert_equals(
        sample.get_array("lp_var_basis_status"),
        np.array(["U", "B", "U", "L", "U"], dtype="S"),
    )
    assert_equals(
        sample.get_array("lp_var_reduced_costs"),
        [193.615385, 0.0, 187.230769, -23.692308, 13.538462],
    )
    assert_equals(
        sample.get_array("lp_var_sa_lb_down"),
        [-inf, -inf, -inf, -0.111111, -inf],
    )
    assert_equals(
        sample.get_array("lp_var_sa_lb_up"),
        [1.0, 0.923077, 1.0, 1.0, 67.0],
    )
    assert_equals(
        sample.get_array("lp_var_sa_obj_down"),
        [311.384615, 317.777778, 270.769231, -inf, -13.538462],
    )
    assert_equals(
        sample.get_array("lp_var_sa_obj_up"),
        [inf, 570.869565, inf, 243.692308, inf],
    )
    assert_equals(
        sample.get_array("lp_var_sa_ub_down"),
        np.array([0.913043, 0.923077, 0.9, 0.0, 43.0]),
    )
    assert_equals(
        sample.get_array("lp_var_sa_ub_up"),
        np.array([2.043478, inf, 2.2, inf, 69.0]),
    )
    assert_equals(
        sample.get_array("lp_var_values"),
        np.array([1.0, 0.923077, 1.0, 0.0, 67.0]),
    )
    assert_equals(
        sample.get_array("lp_var_features"),
        np.array(
            [
                [
                    23.0,
                    505.0,
                    1.0,
                    0.32899,
                    0.0,
                    505.0,
                    1.0,
                    1.0,
                    0.32899,
                    0.0,
                    1.0,
                    1.0,
                    5.265874,
                    0.0,
                    193.615385,
                    -0.111111,
                    1.0,
                    311.384615,
                    570.869565,
                    0.913043,
                    2.043478,
                    1.0,
                ],
                [
                    26.0,
                    352.0,
                    1.0,
                    0.229316,
                    0.0,
                    352.0,
                    1.0,
                    1.0,
                    0.229316,
                    0.076923,
                    1.0,
                    1.0,
                    3.532875,
                    0.0,
                    0.0,
                    -0.111111,
                    0.923077,
                    317.777778,
                    570.869565,
                    0.923077,
                    69.0,
                    0.923077,
                ],
                [
                    20.0,
                    458.0,
                    1.0,
                    0.298371,
                    0.0,
                    458.0,
                    1.0,
                    1.0,
                    0.298371,
                    0.0,
                    1.0,
                    1.0,
                    5.232342,
                    0.0,
                    187.230769,
                    -0.111111,
                    1.0,
                    270.769231,
                    570.869565,
                    0.9,
                    2.2,
                    1.0,
                ],
                [
                    18.0,
                    220.0,
                    1.0,
                    0.143322,
                    0.0,
                    220.0,
                    1.0,
                    1.0,
                    0.143322,
                    0.0,
                    1.0,
                    -1.0,
                    5.453347,
                    0.0,
                    -23.692308,
                    -0.111111,
                    1.0,
                    -13.538462,
                    243.692308,
                    0.0,
                    69.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    67.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    5.453347,
                    0.0,
                    13.538462,
                    -0.111111,
                    67.0,
                    -13.538462,
                    570.869565,
                    43.0,
                    69.0,
                    67.0,
                ],
            ]
        ),
    )
    assert_equals(
        sample.get_array("lp_constr_basis_status"),
        np.array(["N"], dtype="S"),
    )
    assert_equals(
        sample.get_array("lp_constr_dual_values"),
        np.array([13.538462]),
    )
    assert_equals(
        sample.get_array("lp_constr_sa_rhs_down"),
        np.array([-24.0]),
    )
    assert_equals(
        sample.get_array("lp_constr_sa_rhs_up"),
        np.array([2.0]),
    )
    assert_equals(
        sample.get_array("lp_constr_slacks"),
        np.array([0.0]),
    )
    assert_equals(
        sample.get_array("lp_constr_features"),
        np.array([[0.0, 13.538462, -24.0, 2.0, 0.0]]),
    )

    # after-mip
    # -------------------------------------------------------
    solver.solve()
    extractor.extract_after_mip_features(solver, sample)
    assert_equals(
        sample.get_array("mip_var_values"), np.array([1.0, 0.0, 1.0, 1.0, 61.0])
    )
    assert_equals(sample.get_array("mip_constr_slacks"), np.array([0.0]))


def test_constraint_getindex() -> None:
    cf = Constraints(
        names=np.array(["c1", "c2", "c3"], dtype="S"),
        rhs=np.array([1.0, 2.0, 3.0]),
        senses=np.array(["=", "<", ">"], dtype="S"),
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
            names=np.array(["c1", "c3"], dtype="S"),
            rhs=np.array([1.0, 3.0]),
            senses=np.array(["=", ">"], dtype="S"),
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
    lp_stats = solver.solve_lp(tee=True)
    extractor = FeaturesExtractor(with_lhs=False)
    sample = Hdf5Sample("tmp/prof.h5", mode="w")

    def run() -> None:
        extractor.extract_after_load_features(instance, solver, sample)
        extractor.extract_after_lp_features(solver, sample, lp_stats)

    cProfile.run("run()", filename="tmp/prof")
    os.system("flameprof tmp/prof > tmp/prof.svg")
