#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from miplearn.features import ModelFeaturesExtractor
from tests.fixtures.knapsack import get_knapsack_instance
from tests.solvers import get_internal_solvers


def test_knapsack() -> None:
    for solver_factory in get_internal_solvers():
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
        assert features["ConstraintRHS"]["eq_capacity"] == 67.0
