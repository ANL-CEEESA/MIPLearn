#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import GurobiSolver
from miplearn.features import ModelFeaturesExtractor
from tests.fixtures.knapsack import get_knapsack_instance


def test_knapsack() -> None:
    for solver_factory in [GurobiSolver]:
        # Initialize model, instance and internal solver
        solver = solver_factory()
        instance = get_knapsack_instance(solver)
        model = instance.to_model()
        solver.set_instance(instance, model)

        # Extract all model features
        extractor = ModelFeaturesExtractor(solver)
        features = extractor.extract()

        # Test constraint features
        print(solver, features)
        assert features["Variables"] == {
            "x": {
                0: None,
                1: None,
                2: None,
                3: None,
            }
        }
        assert features["Constraints"]["eq_capacity"]["LHS"] == {
            "x[0]": 23.0,
            "x[1]": 26.0,
            "x[2]": 20.0,
            "x[3]": 18.0,
        }
        assert features["Constraints"]["eq_capacity"]["Sense"] == "<"
        assert features["Constraints"]["eq_capacity"]["RHS"] == 67.0
