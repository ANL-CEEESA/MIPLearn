#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import json
import logging
from typing import Any, List, Dict

import gurobipy
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
from miplearn.types import ConstraintName

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
    def find_violated_user_cuts(self, model: Any) -> Dict[ConstraintName, Any]:
        assert isinstance(model, gp.Model)
        try:
            vals = model.cbGetNodeRel(model.getVars())
        except gurobipy.GurobiError:
            return {}
        violations = {}
        for clique in nx.find_cliques(self.graph):
            if sum(vals[i] for i in clique) > 1:
                vname = (",".join([str(i) for i in clique])).encode()
                violations[vname] = list(clique)
        return violations

    @overrides
    def enforce_user_cut(
        self,
        solver: GurobiSolver,
        model: Any,
        clique: List[int],
    ) -> Any:
        x = model.getVars()
        constr = gp.quicksum([x[i] for i in clique]) <= 1
        if solver.cb_where:
            model.cbCut(constr)
        else:
            model.addConstr(constr)


@pytest.fixture
def stab_instance() -> Instance:
    graph = nx.generators.random_graphs.binomial_graph(50, 0.50, seed=42)
    return GurobiStableSetProblem(graph)


@pytest.fixture
def solver() -> LearningSolver:
    return LearningSolver(
        solver=GurobiSolver(params={"Threads": 1}),
        components=[UserCutsComponent()],
    )


def test_usage(
    stab_instance: Instance,
    solver: LearningSolver,
) -> None:
    stats_before = solver.solve(stab_instance)
    sample = stab_instance.get_samples()[0]
    user_cuts_encoded = sample.get_scalar("mip_user_cuts")
    assert user_cuts_encoded is not None
    user_cuts = json.loads(user_cuts_encoded)
    assert user_cuts is not None
    assert len(user_cuts) > 0
    assert stats_before["UserCuts: Added ahead-of-time"] == 0
    assert stats_before["UserCuts: Added in callback"] > 0

    solver.fit([stab_instance])
    stats_after = solver.solve(stab_instance)
    assert (
        stats_after["UserCuts: Added ahead-of-time"]
        == stats_before["UserCuts: Added in callback"]
    )
