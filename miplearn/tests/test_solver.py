# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver
from miplearn.problems.knapsack import MultiKnapsackInstance
from miplearn.branching import BranchPriorityComponent
from miplearn.warmstart import WarmStartComponent
import numpy as np


def _get_instance():
    return MultiKnapsackInstance(
        weights=np.array([[23., 26., 20., 18.]]),
        prices=np.array([505., 352., 458., 220.]),
        capacities=np.array([67.])
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
    assert len(solver.components["warm-start"].x_train[0]) == 10
    assert len(solver.components["warm-start"].y_train[0]) == 10
    
def test_solver_random_branch_priority():
    instance = _get_instance()
    components = {
        "warm-start": BranchPriorityComponent(initial_priority=np.array([1, 2, 3, 4])),
    }
    solver = LearningSolver(components=components)
    solver.solve(instance)
    solver.fit()