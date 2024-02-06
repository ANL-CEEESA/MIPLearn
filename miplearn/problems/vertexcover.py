#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from networkx import Graph
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from .stab import MaxWeightStableSetGenerator
from miplearn.solvers.gurobi import GurobiModel
from ..io import read_pkl_gz


@dataclass
class MinWeightVertexCoverData:
    graph: Graph
    weights: np.ndarray


class MinWeightVertexCoverGenerator:
    def __init__(
        self,
        w: rv_frozen = uniform(loc=10.0, scale=1.0),
        n: rv_frozen = randint(low=250, high=251),
        p: rv_frozen = uniform(loc=0.05, scale=0.0),
        fix_graph: bool = True,
    ):
        self._generator = MaxWeightStableSetGenerator(w, n, p, fix_graph)

    def generate(self, n_samples: int) -> List[MinWeightVertexCoverData]:
        return [
            MinWeightVertexCoverData(s.graph, s.weights)
            for s in self._generator.generate(n_samples)
        ]


def build_vertexcover_model_gurobipy(
    data: Union[str, MinWeightVertexCoverData]
) -> GurobiModel:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, MinWeightVertexCoverData)
    model = gp.Model()
    nodes = list(data.graph.nodes)
    x = model.addVars(nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(quicksum(data.weights[i] * x[i] for i in nodes))
    for v1, v2 in data.graph.edges:
        model.addConstr(x[v1] + x[v2] >= 1)
    model.update()
    return GurobiModel(model)
