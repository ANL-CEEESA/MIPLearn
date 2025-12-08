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

from .stab import (
    MaxWeightStableSetGenerator,
    MaxWeightStableSetPerturber,
    MaxWeightStableSetData,
)
from miplearn.solvers.gurobi import GurobiModel
from ..io import read_pkl_gz


@dataclass
class MinWeightVertexCoverData:
    graph: Graph
    weights: np.ndarray


class MinWeightVertexCoverGenerator:
    """Random instance generator for the Minimum-Weight Vertex Cover Problem.

    Generates instances by creating a new random Erdős-Rényi graph $G_{n,p}$ for each
    instance, where $n$ and $p$ are sampled from user-provided probability distributions
    `n` and `p`. For each instance, the generator independently samples each $w_v$ from
    the user-provided probability distribution `w`.
    """

    def __init__(
        self,
        w: rv_frozen = uniform(loc=10.0, scale=1.0),
        n: rv_frozen = randint(low=250, high=251),
        p: rv_frozen = uniform(loc=0.05, scale=0.0),
    ):
        """Initialize the problem generator.

        Parameters
        ----------
        w: rv_continuous
            Probability distribution for vertex weights.
        n: rv_discrete
            Probability distribution for parameter $n$ in Erdős-Rényi model.
        p: rv_continuous
            Probability distribution for parameter $p$ in Erdős-Rényi model.
        """
        assert isinstance(w, rv_frozen), "w should be a SciPy probability distribution"
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(p, rv_frozen), "p should be a SciPy probability distribution"
        self._generator = MaxWeightStableSetGenerator(w, n, p)

    def generate(self, n_samples: int) -> List[MinWeightVertexCoverData]:
        return [
            MinWeightVertexCoverData(s.graph, s.weights)
            for s in self._generator.generate(n_samples)
        ]


class MinWeightVertexCoverPerturber:
    """Perturbation generator for existing Minimum-Weight Vertex Cover instances.

    Takes an existing MinWeightVertexCoverData instance and generates new instances
    by applying randomization factors to the existing weights while keeping the graph fixed.
    """

    def __init__(
        self,
        w_jitter: rv_frozen = uniform(loc=0.9, scale=0.2),
    ):
        """Initialize the perturbation generator.

        Parameters
        ----------
        w_jitter: rv_continuous
            Probability distribution for randomization factors applied to vertex weights.
        """
        self._perturber = MaxWeightStableSetPerturber(w_jitter)

    def perturb(
        self,
        instance: MinWeightVertexCoverData,
        n_samples: int,
    ) -> List[MinWeightVertexCoverData]:
        stab_instance = MaxWeightStableSetData(instance.graph, instance.weights)
        perturbed_instances = self._perturber.perturb(stab_instance, n_samples)
        return [
            MinWeightVertexCoverData(s.graph, s.weights) for s in perturbed_instances
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
