#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn.features import (
    FeaturesExtractor,
    InstanceFeatures,
    VariableFeatures,
    ConstraintFeatures,
)
from miplearn.solvers.gurobi import GurobiSolver
from tests.fixtures.knapsack import get_knapsack_instance


def test_knapsack() -> None:
    for solver_factory in [GurobiSolver]:
        solver = solver_factory()
        instance = get_knapsack_instance(solver)
        model = instance.to_model()
        solver.set_instance(instance, model)
        FeaturesExtractor(solver).extract(instance)
        assert instance.features.variables == {
            "x[0]": VariableFeatures(
                category="default",
                user_features=[23.0, 505.0],
            ),
            "x[1]": VariableFeatures(
                category="default",
                user_features=[26.0, 352.0],
            ),
            "x[2]": VariableFeatures(
                category="default",
                user_features=[20.0, 458.0],
            ),
            "x[3]": VariableFeatures(
                category="default",
                user_features=[18.0, 220.0],
            ),
        }
        assert instance.features.constraints == {
            "eq_capacity": ConstraintFeatures(
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
        }
        assert instance.features.instance == InstanceFeatures(
            user_features=[67.0, 21.75],
            lazy_constraint_count=0,
        )
