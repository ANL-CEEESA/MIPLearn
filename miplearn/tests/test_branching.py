# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import BranchPriorityComponent, LearningSolver
from miplearn.problems.knapsack import MultiKnapsackInstance
import numpy as np
import tempfile

def _get_instances():
    return [
        MultiKnapsackInstance(
            weights=np.array([[23., 26., 20., 18.]]),
            prices=np.array([505., 352., 458., 220.]),
            capacities=np.array([67.])
        ),
    ] * 2


def test_branching():
    instances = _get_instances()
    component = BranchPriorityComponent()
    for instance in instances:
        component.after_solve(None, instance, None)
    component.fit(None)
    for key in [0, 1, 2, 3]:
        assert key in component.x_train.keys()
        assert key in component.y_train.keys()
        assert component.x_train[key].shape == (2,  9)
        assert component.y_train[key].shape == (2,  1)
        
        
def test_branch_priority_save_load():
    state_file = tempfile.NamedTemporaryFile(mode="r")
    solver = LearningSolver(components={"branch-priority": BranchPriorityComponent()})
    solver.parallel_solve(_get_instances(), n_jobs=2)
    solver.fit()
    comp = solver.components["branch-priority"]
    assert comp.x_train[0].shape == (2, 9)
    assert comp.y_train[0].shape == (2, 1)
    assert 0 in comp.predictors.keys()
    solver.save_state(state_file.name)
    
    solver = LearningSolver(components={"branch-priority": BranchPriorityComponent()})
    solver.load_state(state_file.name)
    comp = solver.components["branch-priority"]
    assert comp.x_train[0].shape == (2, 9)
    assert comp.y_train[0].shape == (2, 1)
    assert 0 in comp.predictors.keys()
