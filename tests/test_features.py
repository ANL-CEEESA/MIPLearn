#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import GurobiSolver
from miplearn.features import FeaturesExtractor
from tests.fixtures.knapsack import get_knapsack_instance


def test_knapsack() -> None:
    for solver_factory in [GurobiSolver]:
        solver = solver_factory()
        instance = get_knapsack_instance(solver)
        model = instance.to_model()
        solver.set_instance(instance, model)
        FeaturesExtractor(solver).extract(instance)
        assert instance.features.variables == {
            "x": {
                0: {
                    "Category": "default",
                    "User features": [23.0, 505.0],
                },
                1: {
                    "Category": "default",
                    "User features": [26.0, 352.0],
                },
                2: {
                    "Category": "default",
                    "User features": [20.0, 458.0],
                },
                3: {
                    "Category": "default",
                    "User features": [18.0, 220.0],
                },
            }
        }
        assert instance.features.constraints == {
            "eq_capacity": {
                "LHS": {
                    "x[0]": 23.0,
                    "x[1]": 26.0,
                    "x[2]": 20.0,
                    "x[3]": 18.0,
                },
                "Sense": "<",
                "RHS": 67.0,
                "Lazy": False,
                "Category": "eq_capacity",
                "User features": [0.0],
            }
        }
        assert instance.features.instance == {
            "User features": [67.0, 21.75],
            "Lazy constraint count": 0,
        }
