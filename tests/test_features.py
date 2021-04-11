#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn.features import (
    FeaturesExtractor,
    InstanceFeatures,
    Variable,
    Constraint,
)
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.tests import assert_equals


def test_knapsack() -> None:
    solver = GurobiSolver()
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    FeaturesExtractor(solver).extract(instance)
    assert_equals(
        instance.features.variables,
        {
            "x[0]": Variable(
                category="default",
                lower_bound=0.0,
                obj_coeff=505.0,
                type="B",
                upper_bound=1.0,
                user_features=[23.0, 505.0],
            ),
            "x[1]": Variable(
                category="default",
                lower_bound=0.0,
                obj_coeff=352.0,
                type="B",
                upper_bound=1.0,
                user_features=[26.0, 352.0],
            ),
            "x[2]": Variable(
                category="default",
                lower_bound=0.0,
                obj_coeff=458.0,
                type="B",
                upper_bound=1.0,
                user_features=[20.0, 458.0],
            ),
            "x[3]": Variable(
                category="default",
                lower_bound=0.0,
                obj_coeff=220.0,
                type="B",
                upper_bound=1.0,
                user_features=[18.0, 220.0],
            ),
        },
    )
    assert_equals(
        instance.features.constraints,
        {
            "eq_capacity": Constraint(
                lhs={
                    "x[0]": 23.0,
                    "x[1]": 26.0,
                    "x[2]": 20.0,
                    "x[3]": 18.0,
                },
                sense="<",
                rhs=67.0,
                lazy=False,
                category="eq_capacity",
                user_features=[0.0],
            )
        },
    )
    assert_equals(
        instance.features.instance,
        InstanceFeatures(
            user_features=[67.0, 21.75],
            lazy_constraint_count=0,
        ),
    )
