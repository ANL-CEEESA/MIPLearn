#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import ObjectiveValueComponent, LearningSolver
from miplearn.problems.knapsack import KnapsackInstance

def _get_instances():
    instances = [
        KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        ),
    ]
    models = [instance.to_model() for instance in instances]
    solver = LearningSolver()
    for i in range(len(instances)):
        solver.solve(instances[i], models[i])
    return instances, models


def test_usage():
    instances, models = _get_instances()
    comp = ObjectiveValueComponent()
    comp.fit(instances)
    assert instances[0].lower_bound == 1183.0
    assert instances[0].upper_bound == 1183.0
    assert comp.predict(instances).tolist() == [[1183.0, 1183.0]]
