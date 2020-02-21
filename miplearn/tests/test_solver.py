# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright © 2020, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver, BranchPriorityComponent, WarmStartComponent
from miplearn.problems.knapsack import KnapsackInstance
import numpy as np


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
    solver.fit()
    solver.solve(instance)

def test_solve_save_load_state():
    instance = _get_instance()
    components_before = {
        "warm-start": WarmStartComponent(),
        "branch-priority": BranchPriorityComponent(),
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
    
def test_solver_random_branch_priority():
    instance = _get_instance()
    components = {
        "branch-priority": BranchPriorityComponent(),
    }
    solver = LearningSolver(components=components)
    solver.solve(instance)
    solver.fit()