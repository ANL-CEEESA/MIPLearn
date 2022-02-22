#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import networkx as nx
import numpy as np
from scipy.stats import uniform, randint

from miplearn.problems.stab import MaxWeightStableSetInstance
from miplearn.solvers.learning import LearningSolver


def test_stab() -> None:
    graph = nx.cycle_graph(5)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    instance = MaxWeightStableSetInstance(graph, weights)
    solver = LearningSolver()
    stats = solver.solve(instance)
    assert stats["mip_lower_bound"] == 2.0


def test_stab_generator_fixed_graph() -> None:
    np.random.seed(42)
    from miplearn.problems.stab import MaxWeightStableSetGenerator

    gen = MaxWeightStableSetGenerator(
        w=uniform(loc=50.0, scale=10.0),
        n=randint(low=10, high=11),
        p=uniform(loc=0.05, scale=0.0),
        fix_graph=True,
    )
    data = gen.generate(1_000)
    weights = np.array([d.weights for d in data])
    weights_avg_actual = np.round(np.average(weights, axis=0))
    weights_avg_expected = [55.0] * 10
    assert list(weights_avg_actual) == weights_avg_expected


def test_stab_generator_random_graph() -> None:
    np.random.seed(42)
    from miplearn.problems.stab import MaxWeightStableSetGenerator

    gen = MaxWeightStableSetGenerator(
        w=uniform(loc=50.0, scale=10.0),
        n=randint(low=30, high=41),
        p=uniform(loc=0.5, scale=0.0),
        fix_graph=False,
    )
    data = gen.generate(1_000)
    n_nodes = [d.graph.number_of_nodes() for d in data]
    n_edges = [d.graph.number_of_edges() for d in data]
    assert np.round(np.mean(n_nodes)) == 35.0
    assert np.round(np.mean(n_edges), -1) == 300.0
