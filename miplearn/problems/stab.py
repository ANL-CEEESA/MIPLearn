#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List, Dict, Hashable

import networkx as nx
import numpy as np
import pyomo.environ as pe
from networkx import Graph
from overrides import overrides
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.instance.base import Instance
from miplearn.types import VariableName, Category


class ChallengeA:
    def __init__(
        self,
        seed: int = 42,
        n_training_instances: int = 500,
        n_test_instances: int = 50,
    ) -> None:
        np.random.seed(seed)
        self.generator = MaxWeightStableSetGenerator(
            w=uniform(loc=100.0, scale=50.0),
            n=randint(low=200, high=201),
            p=uniform(loc=0.05, scale=0.0),
            fix_graph=True,
        )

        np.random.seed(seed + 1)
        self.training_instances = self.generator.generate(n_training_instances)

        np.random.seed(seed + 2)
        self.test_instances = self.generator.generate(n_test_instances)


class MaxWeightStableSetInstance(Instance):
    """An instance of the Maximum-Weight Stable Set Problem.

    Given a graph G=(V,E) and a weight w_v for each vertex v, the problem asks for a stable
    set S of G maximizing sum(w_v for v in S). A stable set (also called independent set) is
    a subset of vertices, no two of which are adjacent.

    This is one of Karp's 21 NP-complete problems.
    """

    def __init__(self, graph: Graph, weights: np.ndarray) -> None:
        super().__init__()
        self.graph = graph
        self.weights = weights
        self.nodes = list(self.graph.nodes)

    @overrides
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        model.x = pe.Var(self.nodes, domain=pe.Binary)
        model.OBJ = pe.Objective(
            expr=sum(model.x[v] * self.weights[v] for v in self.nodes),
            sense=pe.maximize,
        )
        model.clique_eqs = pe.ConstraintList()
        for clique in nx.find_cliques(self.graph):
            model.clique_eqs.add(sum(model.x[v] for v in clique) <= 1)
        return model

    @overrides
    def get_variable_features(self) -> Dict[str, List[float]]:
        features = {}
        for v1 in self.nodes:
            neighbor_weights = [0.0] * 15
            neighbor_degrees = [100.0] * 15
            for v2 in self.graph.neighbors(v1):
                neighbor_weights += [self.weights[v2] / self.weights[v1]]
                neighbor_degrees += [self.graph.degree(v2) / self.graph.degree(v1)]
            neighbor_weights.sort(reverse=True)
            neighbor_degrees.sort()
            f = []
            f += neighbor_weights[:5]
            f += neighbor_degrees[:5]
            f += [self.graph.degree(v1)]
            features[f"x[{v1}]"] = f
        return features

    @overrides
    def get_variable_categories(self) -> Dict[str, Hashable]:
        return {f"x[{v}]": "default" for v in self.nodes}


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

    def generate(self, n_samples: int) -> List[MaxWeightStableSetInstance]:
        def _sample() -> MaxWeightStableSetInstance:
            if self.graph is not None:
                graph = self.graph
            else:
                graph = self._generate_graph()
            weights = self.w.rvs(graph.number_of_nodes())
            return MaxWeightStableSetInstance(graph, weights)

        return [_sample() for _ in range(n_samples)]

    def _generate_graph(self) -> Graph:
        return nx.generators.random_graphs.binomial_graph(self.n.rvs(), self.p.rvs())
