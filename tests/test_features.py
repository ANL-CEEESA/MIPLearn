#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np

from miplearn.features import (
    FeaturesExtractor,
    InstanceFeatures,
    VariableFeatures,
    ConstraintFeatures,
)
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.tests import assert_equals

inf = float("inf")


def test_knapsack() -> None:
    solver = GurobiSolver()
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    solver.solve_lp()

    features = FeaturesExtractor().extract(instance, solver)
    assert features.variables is not None
    assert features.instance is not None

    assert_equals(
        features.variables,
        VariableFeatures(
            names=["x[0]", "x[1]", "x[2]", "x[3]", "z"],
            basis_status=["U", "B", "U", "L", "U"],
            categories=["default", "default", "default", "default", None],
            lower_bounds=[0.0, 0.0, 0.0, 0.0, 0.0],
            obj_coeffs=[505.0, 352.0, 458.0, 220.0, 0.0],
            reduced_costs=[193.615385, 0.0, 187.230769, -23.692308, 13.538462],
            sa_lb_down=[-inf, -inf, -inf, -0.111111, -inf],
            sa_lb_up=[1.0, 0.923077, 1.0, 1.0, 67.0],
            sa_obj_down=[311.384615, 317.777778, 270.769231, -inf, -13.538462],
            sa_obj_up=[inf, 570.869565, inf, 243.692308, inf],
            sa_ub_down=[0.913043, 0.923077, 0.9, 0.0, 43.0],
            sa_ub_up=[2.043478, inf, 2.2, inf, 69.0],
            types=["B", "B", "B", "B", "C"],
            upper_bounds=[1.0, 1.0, 1.0, 1.0, 67.0],
            user_features=[
                [23.0, 505.0],
                [26.0, 352.0],
                [20.0, 458.0],
                [18.0, 220.0],
                None,
            ],
            values=[1.0, 0.923077, 1.0, 0.0, 67.0],
            alvarez_2017=[
                [1.0, 0.32899, 0.0, 0.0, 1.0, 1.0, 5.265874, 46.051702],
                [1.0, 0.229316, 0.0, 0.076923, 1.0, 1.0, 3.532875, 5.388476],
                [1.0, 0.298371, 0.0, 0.0, 1.0, 1.0, 5.232342, 46.051702],
                [1.0, 0.143322, 0.0, 0.0, 1.0, -1.0, 46.051702, 3.16515],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            ],
        ),
    )
    assert_equals(
        features.constraints,
        ConstraintFeatures(
            basis_status=["N"],
            categories=["eq_capacity"],
            dual_values=[13.538462],
            names=["eq_capacity"],
            lazy=[False],
            lhs=[
                [
                    ("x[0]", 23.0),
                    ("x[1]", 26.0),
                    ("x[2]", 20.0),
                    ("x[3]", 18.0),
                    ("z", -1.0),
                ],
            ],
            rhs=[0.0],
            sa_rhs_down=[-24.0],
            sa_rhs_up=[2.0],
            senses=["="],
            slacks=[0.0],
            user_features=[None],
        ),
    )
    assert_equals(
        features.instance,
        InstanceFeatures(
            user_features=[67.0, 21.75],
            lazy_constraint_count=0,
        ),
    )


def test_constraint_getindex() -> None:
    cf = ConstraintFeatures(
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
        ConstraintFeatures(
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
    assert_equals(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
    )
    assert_equals(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert_equals(
        VariableFeatures(values=np.array([1.0, 2.0])),  # type: ignore
        VariableFeatures(values=np.array([1.0, 2.0])),  # type: ignore
    )
    assert_equals(
        np.array([True, True]),
        [True, True],
    )
    assert_equals((1.0,), (1.0,))
    assert_equals({"x": 10}, {"x": 10})
