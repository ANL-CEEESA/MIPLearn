#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Any, FrozenSet, List

import gurobipy as gp
import networkx as nx
import pytest
from gurobipy import GRB
from networkx import Graph
from overrides import overrides

from miplearn.solvers.learning import InternalSolver
from miplearn.components.dynamic_user_cuts import UserCutsComponent
from miplearn.instance.base import Instance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.learning import LearningSolver
from miplearn.types import ConstraintName, ConstraintCategory

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
    def find_violated_user_cuts(self, model: Any) -> List[ConstraintName]:
        assert isinstance(model, gp.Model)
        vals = model.cbGetNodeRel(model.getVars())
        violations = []
        for clique in nx.find_cliques(self.graph):
            if sum(vals[i] for i in clique) > 1:
                violations.append(",".join([str(i) for i in clique]).encode())
        return violations

    @overrides
    def enforce_user_cut(
        self,
        solver: InternalSolver,
        model: Any,
        cid: ConstraintName,
    ) -> Any:
        clique = [int(i) for i in cid.decode().split(",")]
        x = model.getVars()
        model.addConstr(gp.quicksum([x[i] for i in clique]) <= 1)


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
    sample = stab_instance.get_samples()[0]
    user_cuts_enforced = sample.get_set("mip_user_cuts_enforced")
    assert user_cuts_enforced is not None
    assert len(user_cuts_enforced) > 0
    assert stats_before["UserCuts: Added ahead-of-time"] == 0
    assert stats_before["UserCuts: Added in callback"] > 0

    solver.fit([stab_instance])
    stats_after = solver.solve(stab_instance)
    assert (
        stats_after["UserCuts: Added ahead-of-time"]
        == stats_before["UserCuts: Added in callback"]
    )
