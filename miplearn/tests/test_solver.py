# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn import LearningSolver
from miplearn.problems.knapsack import KnapsackInstance2


def test_solver():
    instance = KnapsackInstance2(weights=[23., 26., 20., 18.],
                                 prices=[505., 352., 458., 220.],
                                 capacity=67.)
    solver = LearningSolver()
    solver.solve(instance)
    solver.fit()
    solver.solve(instance)
