#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import BranchPriorityComponent, LearningSolver
from miplearn.problems.knapsack import KnapsackInstance
import numpy as np
import tempfile

def _get_instances():
    return [
        KnapsackInstance(
            weights=[23., 26., 20., 18.],
            prices=[505., 352., 458., 220.],
            capacity=67.,
        ),
    ] * 2


def test_branching():
    instances = _get_instances()
    component = BranchPriorityComponent()
    for instance in instances:
        component.after_solve(None, instance, None)
    component.fit(None)
    for key in ["default"]:
        assert key in component.x_train.keys()
        assert key in component.y_train.keys()
        assert component.x_train[key].shape == (8,  4)
        assert component.y_train[key].shape == (8,  1)
        
        
def test_branch_priority_save_load():
    state_file = tempfile.NamedTemporaryFile(mode="r")
    solver = LearningSolver(components={"branch-priority": BranchPriorityComponent()})
    solver.parallel_solve(_get_instances(), n_jobs=2)
    solver.fit()
    comp = solver.components["branch-priority"]
    assert comp.x_train["default"].shape == (8, 4)
    assert comp.y_train["default"].shape == (8, 1)
    assert "default" in comp.predictors.keys()
    solver.save_state(state_file.name)
    
    solver = LearningSolver(components={"branch-priority": BranchPriorityComponent()})
    solver.load_state(state_file.name)
    comp = solver.components["branch-priority"]
    assert comp.x_train["default"].shape == (8, 4)
    assert comp.y_train["default"].shape == (8, 1)
    assert "default" in comp.predictors.keys()
