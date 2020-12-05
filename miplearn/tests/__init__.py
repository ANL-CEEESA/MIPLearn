#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from miplearn import LearningSolver
from miplearn.problems.knapsack import KnapsackInstance


def get_test_pyomo_instances():
    instances = [
        KnapsackInstance(
            weights=[23.0, 26.0, 20.0, 18.0],
            prices=[505.0, 352.0, 458.0, 220.0],
            capacity=67.0,
        ),
        KnapsackInstance(
            weights=[25.0, 30.0, 22.0, 18.0],
            prices=[500.0, 365.0, 420.0, 150.0],
            capacity=70.0,
        ),
    ]
    models = [instance.to_model() for instance in instances]
    solver = LearningSolver()
    for i in range(len(instances)):
        solver.solve(instances[i], models[i])
    return instances, models
