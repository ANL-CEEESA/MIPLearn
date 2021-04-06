#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Any, FrozenSet, Hashable

import gurobipy as gp
import networkx as nx
import pytest
from gurobipy import GRB
from networkx import Graph

from miplearn import Instance, LearningSolver, GurobiSolver
from miplearn.components.user_cuts import UserCutsComponentNG

logger = logging.getLogger(__name__)


class GurobiStableSetProblem(Instance):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph
        self.nodes = list(self.graph.nodes)

    def to_model(self) -> Any:
        model = gp.Model()
        x = [model.addVar(vtype=GRB.BINARY) for _ in range(len(self.nodes))]
        model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
        for e in list(self.graph.edges):
            model.addConstr(x[e[0]] + x[e[1]] <= 1)
        return model

    def has_user_cuts(self) -> bool:
        return True

    def find_violated_user_cuts(self, model):
        assert isinstance(model, gp.Model)
        vals = model.cbGetNodeRel(model.getVars())
        violations = []
        for clique in nx.find_cliques(self.graph):
            lhs = sum(vals[i] for i in clique)
            if lhs > 1:
                violations += [frozenset(clique)]
        return violations

    def build_user_cut(self, model: Any, violation: Hashable) -> Any:
        assert isinstance(violation, FrozenSet)
        x = model.getVars()
        cut = gp.quicksum([x[i] for i in violation]) <= 1
        return cut


@pytest.fixture
def stab_instance() -> Instance:
    graph = nx.generators.random_graphs.binomial_graph(50, 0.50, seed=42)
    return GurobiStableSetProblem(graph)


@pytest.fixture
def solver() -> LearningSolver:
    return LearningSolver(
        solver=lambda: GurobiSolver(),
        components=[
            UserCutsComponentNG(),
        ],
    )


def test_usage(
    stab_instance: Instance,
    solver: LearningSolver,
) -> None:
    solver.solve(stab_instance)
    sample = stab_instance.training_data[0]
    assert sample.user_cuts_enforced is not None
    assert len(sample.user_cuts_enforced) > 0
