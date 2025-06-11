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
    """
    Random instance generator for the Maximum Cut Problem.

    The generator operates in two modes. When `fix_graph=True`, a single random
    Erdős-Rényi graph $G_{n,p}$ is generated during initialization, with parameters $n$
    and $p$ drawn from their respective probability distributions. For each instance,
    only edge weights are randomly sampled from the set {1, -1}, while the graph
    structure remains fixed.

    When `fix_graph=False`, both the graph structure and edge weights are randomly
    generated for each instance.
    """

    def __init__(
        self,
        n: rv_frozen,
        p: rv_frozen,
        fix_graph: bool,
    ):
        """
        Initialize the problem generator.

        Parameters
        ----------
        n: rv_discrete
            Probability distribution for the number of nodes.
        p: rv_continuous
            Probability distribution for the graph density.
        fix_graph: bool
            Controls graph generation for instances. If false, a new random graph is
            generated for each instance. If true, the same graph is reused across instances.
        """
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(p, rv_frozen), "p should be a SciPy probability distribution"
        self.n = n
        self.p = p
        self.fix_graph = fix_graph
        self.graph = None
        if fix_graph:
            self.graph = self._generate_graph()

    def generate(self, n_samples: int) -> List[MaxCutData]:
        def _sample() -> MaxCutData:
            if self.graph is not None:
                graph = self.graph
            else:
                graph = self._generate_graph()
            m = graph.number_of_edges()
            weights = np.random.randint(2, size=(m,)) * 2 - 1
            return MaxCutData(graph, weights)

        return [_sample() for _ in range(n_samples)]

    def _generate_graph(self) -> Graph:
        return nx.generators.random_graphs.binomial_graph(self.n.rvs(), self.p.rvs())


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
