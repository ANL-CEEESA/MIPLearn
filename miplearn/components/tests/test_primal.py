#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import LearningSolver, PrimalSolutionComponent
from miplearn.problems.knapsack import KnapsackInstance
import numpy as np
import tempfile


def _get_instances():
    instances = [
        KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        ),
    ] * 5
    models = [inst.to_model() for inst in instances]
    solver = LearningSolver()
    for i in range(len(instances)):
        solver.solve(instances[i], models[i])
    return instances, models


def test_predict():
    instances, models = _get_instances()
    comp = PrimalSolutionComponent()
    comp.fit(instances)
    solution = comp.predict(instances[0], models[0])
    assert models[0].x in solution.keys()
    assert solution[models[0].x][0] == 1
    assert solution[models[0].x][1] == 1
    assert solution[models[0].x][2] == 1
    assert solution[models[0].x][3] == 1

# def test_warm_start_save_load():
#     state_file = tempfile.NamedTemporaryFile(mode="r")
#     solver = LearningSolver(components={"warm-start": WarmStartComponent()})
#     solver.parallel_solve(_get_instances(), n_jobs=2)
#     solver.fit()
#     comp = solver.components["warm-start"]
#     assert comp.x_train["default"].shape == (8, 6)
#     assert comp.y_train["default"].shape == (8, 2)
#     assert ("default", 0) in comp.predictors.keys()
#     assert ("default", 1) in comp.predictors.keys()
#     solver.save_state(state_file.name)
    
#     solver.solve(_get_instances()[0])
    
#     solver = LearningSolver(components={"warm-start": WarmStartComponent()})
#     solver.load_state(state_file.name)
#     comp = solver.components["warm-start"]
#     assert comp.x_train["default"].shape == (8, 6)
#     assert comp.y_train["default"].shape == (8, 2)
#     assert ("default", 0) in comp.predictors.keys()
#     assert ("default", 1) in comp.predictors.keys()
