#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from dataclasses import dataclass
from typing import List, Union, Any, Hashable

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB, quicksum
from networkx import Graph
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel

logger = logging.getLogger(__name__)


@dataclass
class MaxWeightStableSetData:
    graph: Graph
    weights: np.ndarray


class MaxWeightStableSetGenerator:
    """Random instance generator for the Maximum-Weight Stable Set Problem.

    The generator has two modes of operation. When `fix_graph=True` is provided,
    one random Erdős-Rényi graph $G_{n,p}$ is generated in the constructor, where $n$
    and $p$ are sampled from user-provided probability distributions `n` and `p`. To
    generate each instance, the generator independently samples each $w_v$ from the
    user-provided probability distribution `w`.

    When `fix_graph=False`, a new random graph is generated for each instance; the
    remaining parameters are sampled in the same way.
    """

    def __init__(
        self,
        w: rv_frozen = uniform(loc=10.0, scale=1.0),
        n: rv_frozen = randint(low=250, high=251),
        p: rv_frozen = uniform(loc=0.05, scale=0.0),
        fix_graph: bool = True,
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
        self.w = w
        self.n = n
        self.p = p
        self.fix_graph = fix_graph
        self.graph = None
        if fix_graph:
            self.graph = self._generate_graph()

    def generate(self, n_samples: int) -> List[MaxWeightStableSetData]:
        def _sample() -> MaxWeightStableSetData:
            if self.graph is not None:
                graph = self.graph
            else:
                graph = self._generate_graph()
            weights = np.round(self.w.rvs(graph.number_of_nodes()), 2)
            return MaxWeightStableSetData(graph, weights)

        return [_sample() for _ in range(n_samples)]

    def _generate_graph(self) -> Graph:
        return nx.generators.random_graphs.binomial_graph(self.n.rvs(), self.p.rvs())


def build_stab_model(data: MaxWeightStableSetData) -> GurobiModel:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, MaxWeightStableSetData)

    model = gp.Model()
    nodes = list(data.graph.nodes)

    # Variables and objective function
    x = model.addVars(nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(quicksum(-data.weights[i] * x[i] for i in nodes))

    # Edge inequalities
    for (i1, i2) in data.graph.edges:
        model.addConstr(x[i1] + x[i2] <= 1)

    def cuts_separate(m: GurobiModel) -> List[Hashable]:
        # Retrieve optimal fractional solution
        x_val = m.inner.cbGetNodeRel(x)

        # Check that we selected at most one vertex for each
        # clique in the graph (sum <= 1)
        violations: List[Hashable] = []
        for clique in nx.find_cliques(data.graph):
            if sum(x_val[i] for i in clique) > 1.0001:
                violations.append(tuple(sorted(clique)))
        return violations

    def cuts_enforce(m: GurobiModel, violations: List[Any]) -> None:
        logger.info(f"Adding {len(violations)} clique cuts...")
        for clique in violations:
            m.add_constr(quicksum(x[i] for i in clique) <= 1)

    model.update()

    return GurobiModel(
        model,
        cuts_separate=cuts_separate,
        cuts_enforce=cuts_enforce,
    )
