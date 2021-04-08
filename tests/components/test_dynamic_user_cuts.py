#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Any, FrozenSet, Hashable

import gurobipy as gp
import networkx as nx
import pytest
from gurobipy import GRB
from networkx import Graph
from overrides import overrides

from miplearn.components.dynamic_user_cuts import UserCutsComponent
from miplearn.instance.base import Instance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.learning import LearningSolver

logger = logging.getLogger(__name__)


class GurobiStableSetProblem(Instance):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph: Graph = graph

    @overrides
    def to_model(self) -> Any:
        model = gp.Model()
        x = [model.addVar(vtype=GRB.BINARY) for _ in range(len(self.graph.nodes))]
        model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
        for e in list(self.graph.edges):
            model.addConstr(x[e[0]] + x[e[1]] <= 1)
        return model

    @overrides
    def has_user_cuts(self) -> bool:
        return True

    @overrides
    def find_violated_user_cuts(self, model):
        assert isinstance(model, gp.Model)
        vals = model.cbGetNodeRel(model.getVars())
        violations = []
        for clique in nx.find_cliques(self.graph):
            if sum(vals[i] for i in clique) > 1:
                violations += [frozenset(clique)]
        return violations

    @overrides
    def build_user_cut(self, model: Any, cid: Hashable) -> Any:
        assert isinstance(cid, FrozenSet)
        x = model.getVars()
        return gp.quicksum([x[i] for i in cid]) <= 1


@pytest.fixture
def stab_instance() -> Instance:
    graph = nx.generators.random_graphs.binomial_graph(50, 0.50, seed=42)
    return GurobiStableSetProblem(graph)


@pytest.fixture
def solver() -> LearningSolver:
    return LearningSolver(
        solver=GurobiSolver(),
        components=[UserCutsComponent()],
    )


def test_usage(
    stab_instance: Instance,
    solver: LearningSolver,
) -> None:
    stats_before = solver.solve(stab_instance)
    sample = stab_instance.training_data[0]
    assert sample.user_cuts_enforced is not None
    assert len(sample.user_cuts_enforced) > 0
    print(stats_before)
    assert stats_before["UserCuts: Added ahead-of-time"] == 0
    assert stats_before["UserCuts: Added in callback"] > 0

    solver.fit([stab_instance])
    stats_after = solver.solve(stab_instance)
    assert (
        stats_after["UserCuts: Added ahead-of-time"]
        == stats_before["UserCuts: Added in callback"]
    )
