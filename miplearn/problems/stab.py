#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Union

import gurobipy as gp
import networkx as nx
import numpy as np
import pyomo.environ as pe
from gurobipy import GRB, quicksum
from networkx import Graph
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel
from miplearn.solvers.pyomo import PyomoModel


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


def build_stab_model_gurobipy(data: MaxWeightStableSetData) -> GurobiModel:
    data = _read_stab_data(data)
    model = gp.Model()
    nodes = list(data.graph.nodes)
    x = model.addVars(nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(quicksum(-data.weights[i] * x[i] for i in nodes))
    for clique in nx.find_cliques(data.graph):
        model.addConstr(quicksum(x[i] for i in clique) <= 1)
    model.update()
    return GurobiModel(model)


def build_stab_model_pyomo(
    data: MaxWeightStableSetData,
    solver="gurobi_persistent",
) -> PyomoModel:
    data = _read_stab_data(data)
    model = pe.ConcreteModel()
    nodes = pe.Set(initialize=list(data.graph.nodes))
    model.x = pe.Var(nodes, domain=pe.Boolean, name="x")
    model.obj = pe.Objective(expr=sum([-data.weights[i] * model.x[i] for i in nodes]))
    model.clique_eqs = pe.ConstraintList()
    for clique in nx.find_cliques(data.graph):
        model.clique_eqs.add(expr=sum(model.x[i] for i in clique) <= 1)
    return PyomoModel(model, solver)


def _read_stab_data(data: Union[str, MaxWeightStableSetData]) -> MaxWeightStableSetData:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, MaxWeightStableSetData)
    return data
