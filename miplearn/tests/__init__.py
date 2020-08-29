#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from miplearn import LearningSolver
from miplearn.problems.knapsack import KnapsackInstance


def get_test_pyomo_instances():
    instances = [
        KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        ),
        KnapsackInstance(
            weights=[25., 30., 22., 18.],
            prices=[500., 365., 420., 150.],
            capacity=70.,
        ),
    ]
    models = [instance.to_model() for instance in instances]
    solver = LearningSolver()
    for i in range(len(instances)):
        solver.solve(instances[i], models[i])
    return instances, models
