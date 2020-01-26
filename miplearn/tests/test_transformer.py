# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from miplearn.transformers import PerVariableTransformer
from miplearn.problems.knapsack import KnapsackInstance, KnapsackInstance2
import numpy as np
import pyomo.environ as pe


def test_transform():
    transformer = PerVariableTransformer()
    instance = KnapsackInstance(weights=[23., 26., 20., 18.],
                                prices=[505., 352., 458., 220.],
                                capacity=67.)
    model = instance.to_model()
    solver = pe.SolverFactory('gurobi')
    solver.options["threads"] = 1
    solver.solve(model)

    var_split = transformer.split_variables(instance, model)
    var_split_expected = {
        "default": [
            (model.x, 0),
            (model.x, 1),
            (model.x, 2),
            (model.x, 3)
        ]
    }
    assert var_split == var_split_expected
    var_index_pairs = [(model.x, i) for i in range(4)]

    x_actual = transformer.transform_instance(instance, var_index_pairs)
    x_expected = np.array([
        [67., 21.75, 23., 505.],
        [67., 21.75, 26., 352.],
        [67., 21.75, 20., 458.],
        [67., 21.75, 18., 220.],
    ])
    assert x_expected.tolist() == np.round(x_actual, decimals=2).tolist()

    solver.solve(model)
    y_actual = transformer.transform_solution(var_index_pairs)
    y_expected = np.array([
        [0., 1.],
        [1., 0.],
        [0., 1.],
        [0., 1.],
    ])
    assert y_actual.tolist() == y_expected.tolist()


def test_transform_with_categories():
    transformer = PerVariableTransformer()
    instance = KnapsackInstance2(weights=[23., 26., 20., 18.],
                                 prices=[505., 352., 458., 220.],
                                 capacity=67.)
    model = instance.to_model()
    solver = pe.SolverFactory('gurobi')
    solver.options["threads"] = 1
    solver.solve(model)

    var_split = transformer.split_variables(instance, model)
    var_split_expected = {
        0: [(model.x, 0)],
        1: [(model.x, 1)],
        2: [(model.x, 2)],
        3: [(model.x, 3)],
    }
    assert var_split == var_split_expected

    var_index_pairs = var_split[0]
    x_actual = transformer.transform_instance(instance, var_index_pairs)
    x_expected = np.array([
        [23., 26., 20., 18., 505., 352., 458., 220.]
    ])
    assert x_expected.tolist() == np.round(x_actual, decimals=2).tolist()

    solver.solve(model)

    y_actual = transformer.transform_solution(var_index_pairs)
    y_expected = np.array([[0., 1.]])
    assert y_actual.tolist() == y_expected.tolist()
