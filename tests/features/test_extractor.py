#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np

from miplearn.features.extractor import FeaturesExtractor
from miplearn.features.sample import Sample, MemorySample
from miplearn.solvers.internal import Variables, Constraints
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.tests import assert_equals

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
    assert_equals(sample.get_vector("var_names"), ["x[0]", "x[1]", "x[2]", "x[3]", "z"])
    assert_equals(sample.get_vector("var_lower_bounds"), [0.0, 0.0, 0.0, 0.0, 0.0])
    assert_equals(
        sample.get_vector("var_obj_coeffs"), [505.0, 352.0, 458.0, 220.0, 0.0]
    )
    assert_equals(sample.get_vector("var_types"), ["B", "B", "B", "B", "C"])
    assert_equals(sample.get_vector("var_upper_bounds"), [1.0, 1.0, 1.0, 1.0, 67.0])
    assert_equals(
        sample.get_vector("var_categories"),
        ["default", "default", "default", "default", None],
    )
    assert_equals(
        sample.get_vector_list("var_features_user"),
        [[23.0, 505.0], [26.0, 352.0], [20.0, 458.0], [18.0, 220.0], None],
    )
    assert_equals(
        sample.get_vector_list("var_features_AlvLouWeh2017"),
        [
            [1.0, 0.32899, 0.0],
            [1.0, 0.229316, 0.0],
            [1.0, 0.298371, 0.0],
            [1.0, 0.143322, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    assert sample.get_vector_list("var_features") is not None
    assert_equals(sample.get_vector("constr_names"), ["eq_capacity"])
    # assert_equals(
    #     sample.get_vector("constr_lhs"),
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
    assert_equals(sample.get_vector("constr_rhs"), [0.0])
    assert_equals(sample.get_vector("constr_senses"), ["="])
    assert_equals(sample.get_vector("constr_features_user"), [None])
    assert_equals(sample.get_vector("constr_categories"), ["eq_capacity"])
    assert_equals(sample.get_vector("constr_lazy"), [False])
    assert_equals(sample.get_vector("instance_features_user"), [67.0, 21.75])
    assert_equals(sample.get_scalar("static_lazy_count"), 0)

    # after-lp
    # -------------------------------------------------------
    solver.solve_lp()
    extractor.extract_after_lp_features(solver, sample)
    assert_equals(
        sample.get_vector("lp_var_basis_status"),
        ["U", "B", "U", "L", "U"],
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
    assert_equals(
        sample.get_vector_list("lp_var_features_AlvLouWeh2017"),
        [
            [1.0, 0.32899, 0.0, 0.0, 1.0, 1.0, 5.265874, 46.051702],
            [1.0, 0.229316, 0.0, 0.076923, 1.0, 1.0, 3.532875, 5.388476],
            [1.0, 0.298371, 0.0, 0.0, 1.0, 1.0, 5.232342, 46.051702],
            [1.0, 0.143322, 0.0, 0.0, 1.0, -1.0, 46.051702, 3.16515],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        ],
    )
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
        rhs=[1.0, 2.0, 3.0],
        senses=["=", "<", ">"],
        lhs=[
            [
                ("x1", 1.0),
                ("x2", 1.0),
            ],
            [
                ("x2", 2.0),
                ("x3", 2.0),
            ],
            [
                ("x3", 3.0),
                ("x4", 3.0),
            ],
        ],
    )
    assert_equals(
        cf[[True, False, True]],
        Constraints(
            names=["c1", "c3"],
            rhs=[1.0, 3.0],
            senses=["=", ">"],
            lhs=[
                [
                    ("x1", 1.0),
                    ("x2", 1.0),
                ],
                [
                    ("x3", 3.0),
                    ("x4", 3.0),
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
