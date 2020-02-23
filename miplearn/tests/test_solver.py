#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import LearningSolver, BranchPriorityComponent, WarmStartComponent
from miplearn.problems.knapsack import KnapsackInstance


def _get_instance():
    return KnapsackInstance(
        weights=[23., 26., 20., 18.],
        prices=[505., 352., 458., 220.],
        capacity=67.,
    )


def test_solver():
    instance = _get_instance()
    solver = LearningSolver()
    solver.solve(instance)
    assert instance.solution["x"][0] == 1.0
    assert instance.solution["x"][1] == 0.0
    assert instance.solution["x"][2] == 1.0
    assert instance.solution["x"][3] == 1.0
    assert instance.lower_bound == 1183.0
    assert instance.upper_bound == 1183.0
    
    solver.fit()
    solver.solve(instance)


def test_solve_save_load_state():
    instance = _get_instance()
    components_before = {
        "warm-start": WarmStartComponent(),
    }
    solver = LearningSolver(components=components_before)
    solver.solve(instance)
    solver.fit()
    solver.save_state("/tmp/knapsack_train.bin")
    prev_x_train_len = len(solver.components["warm-start"].x_train)
    prev_y_train_len = len(solver.components["warm-start"].y_train)
    
    components_after = {
        "warm-start": WarmStartComponent(),
    }
    solver = LearningSolver(components=components_after)
    solver.load_state("/tmp/knapsack_train.bin")
    assert len(solver.components.keys()) == 1
    assert len(solver.components["warm-start"].x_train) == prev_x_train_len
    assert len(solver.components["warm-start"].y_train) == prev_y_train_len


def test_parallel_solve():
    instances = [_get_instance() for _ in range(10)]
    solver = LearningSolver()
    results = solver.parallel_solve(instances, n_jobs=3)
    assert len(results) == 10
    assert len(solver.components["warm-start"].x_train["default"]) == 40
    assert len(solver.components["warm-start"].y_train["default"]) == 40
    
    for instance in instances:
        assert len(instance.solution["x"].keys()) == 4
    
