#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Any, List

import pytest
from networkx import Graph
import networkx as nx
from scipy.stats import randint

from miplearn import Instance
from miplearn.problems.stab import MaxWeightStableSetGenerator


class GurobiStableSetProblem(Instance):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph

    def to_model(self) -> Any:
        pass


@pytest.fixture
def instance() -> Instance:
    graph = nx.generators.random_graphs.binomial_graph(50, 0.5)
    return GurobiStableSetProblem(graph)


def test_usage(instance: Instance) -> None:
    pass
