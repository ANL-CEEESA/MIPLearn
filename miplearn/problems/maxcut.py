#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2025, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Union, Optional, Any

import gurobipy as gp
import networkx as nx
import numpy as np
import pyomo.environ as pe
from networkx import Graph
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.problems import _gurobipy_set_params, _pyomo_set_params
from miplearn.solvers.gurobi import GurobiModel
from miplearn.solvers.pyomo import PyomoModel


@dataclass
class MaxCutData:
    graph: Graph
    weights: np.ndarray


class MaxCutGenerator:
    """Random instance generator for the Maximum Cut Problem.

    Generates instances by creating a new random Erdős-Rényi graph $G_{n,p}$ for each
    instance, where $n$ and $p$ are sampled from user-provided probability distributions.
    For each instance, the generator assigns random edge weights drawn from the set {-1, 1}
    with equal probability.
    """

    def __init__(
        self,
        n: rv_frozen,
        p: rv_frozen,
    ):
        """
        Initialize the problem generator.

        Parameters
        ----------
        n: rv_discrete
            Probability distribution for the number of nodes.
        p: rv_continuous
            Probability distribution for the graph density.
        """
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(p, rv_frozen), "p should be a SciPy probability distribution"
        self.n = n
        self.p = p

    def generate(self, n_samples: int) -> List[MaxCutData]:
        def _sample() -> MaxCutData:
            graph = self._generate_graph()
            weights = self._generate_weights(graph)
            return MaxCutData(graph, weights)

        return [_sample() for _ in range(n_samples)]

    def _generate_graph(self) -> Graph:
        return nx.generators.random_graphs.binomial_graph(self.n.rvs(), self.p.rvs())

    @staticmethod
    def _generate_weights(graph: Graph) -> np.ndarray:
        m = graph.number_of_edges()
        return np.random.randint(2, size=(m,)) * 2 - 1


class MaxCutPerturber:
    """Perturbation generator for existing Maximum Cut instances.

    Takes an existing MaxCutData instance and generates new instances by randomly
    flipping the sign of each edge weight with a given probability while keeping
    the graph structure fixed.
    """

    def __init__(
        self,
        w_jitter: float = 0.05,
    ):
        """Initialize the perturbation generator.

        Parameters
        ----------
        w_jitter: float
            Probability that each edge weight flips sign (from -1 to 1 or vice versa).
        """
        assert 0.0 <= w_jitter <= 1.0, "w_jitter should be between 0.0 and 1.0"
        self.w_jitter = w_jitter

    def perturb(
        self,
        instance: MaxCutData,
        n_samples: int,
    ) -> List[MaxCutData]:
        def _sample() -> MaxCutData:
            jitter = self._generate_jitter(instance.graph)
            weights = instance.weights * jitter
            return MaxCutData(instance.graph, weights)

        return [_sample() for _ in range(n_samples)]

    def _generate_jitter(self, graph: Graph) -> np.ndarray:
        m = graph.number_of_edges()
        return (np.random.rand(m) >= self.w_jitter).astype(int) * 2 - 1


def build_maxcut_model_gurobipy(
    data: Union[str, MaxCutData],
    params: Optional[dict[str, Any]] = None,
) -> GurobiModel:
    # Initialize model
    model = gp.Model()
    _gurobipy_set_params(model, params)

    # Read data
    data = _maxcut_read(data)
    nodes = list(data.graph.nodes())
    edges = list(data.graph.edges())

    # Add decision variables
    x = model.addVars(nodes, vtype=gp.GRB.BINARY, name="x")

    # Add the objective function
    model.setObjective(
        gp.quicksum(
            -data.weights[i] * x[e[0]] * (1 - x[e[1]]) for (i, e) in enumerate(edges)
        )
    )

    model.update()
    return GurobiModel(model)


def build_maxcut_model_pyomo(
    data: Union[str, MaxCutData],
    solver: str = "gurobi_persistent",
    params: Optional[dict[str, Any]] = None,
) -> PyomoModel:
    # Initialize model
    model = pe.ConcreteModel()

    # Read data
    data = _maxcut_read(data)
    nodes = pe.Set(initialize=list(data.graph.nodes))
    edges = list(data.graph.edges())

    # Add decision variables
    model.x = pe.Var(nodes, domain=pe.Binary, name="x")

    # Add the objective function
    model.obj = pe.Objective(
        expr=pe.quicksum(
            -data.weights[i] * model.x[e[0]]
            + data.weights[i] * model.x[e[0]] * model.x[e[1]]
            for (i, e) in enumerate(edges)
        ),
        sense=pe.minimize,
    )
    model.pprint()
    pm = PyomoModel(model, solver)
    _pyomo_set_params(model, params, solver)
    return pm


def _maxcut_read(data: Union[str, MaxCutData]) -> MaxCutData:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, MaxCutData)
    return data
