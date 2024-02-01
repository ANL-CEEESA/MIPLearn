#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from dataclasses import dataclass
from typing import List, Union, Any, Hashable, Optional

import gurobipy as gp
import networkx as nx
import numpy as np
import pyomo.environ as pe
from gurobipy import GRB, quicksum
from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel
from miplearn.solvers.pyomo import PyomoModel
from networkx import Graph
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from . import _gurobipy_set_params, _pyomo_set_params

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


def build_stab_model_gurobipy(
    data: Union[str, MaxWeightStableSetData],
    params: Optional[dict[str, Any]] = None,
) -> GurobiModel:
    model = gp.Model()
    _gurobipy_set_params(model, params)

    data = _stab_read(data)
    nodes = list(data.graph.nodes)

    # Variables and objective function
    x = model.addVars(nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(quicksum(-data.weights[i] * x[i] for i in nodes))

    # Edge inequalities
    for (i1, i2) in data.graph.edges:
        model.addConstr(x[i1] + x[i2] <= 1)

    def cuts_separate(m: GurobiModel) -> List[Hashable]:
        x_val = m.inner.cbGetNodeRel(x)
        return _stab_separate(data, x_val)

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


def build_stab_model_pyomo(
    data: MaxWeightStableSetData,
    solver: str = "gurobi_persistent",
    params: Optional[dict[str, Any]] = None,
) -> PyomoModel:
    data = _stab_read(data)
    model = pe.ConcreteModel()
    nodes = pe.Set(initialize=list(data.graph.nodes))

    # Variables and objective function
    model.x = pe.Var(nodes, domain=pe.Boolean, name="x")
    model.obj = pe.Objective(expr=sum([-data.weights[i] * model.x[i] for i in nodes]))

    # Edge inequalities
    model.edge_eqs = pe.ConstraintList()
    for (i1, i2) in data.graph.edges:
        model.edge_eqs.add(model.x[i1] + model.x[i2] <= 1)

    # Clique inequalities
    model.clique_eqs = pe.ConstraintList()

    def cuts_separate(m: PyomoModel) -> List[Hashable]:
        m.solver.cbGetNodeRel([model.x[i] for i in nodes])
        x_val = [model.x[i].value for i in nodes]
        return _stab_separate(data, x_val)

    def cuts_enforce(m: PyomoModel, violations: List[Any]) -> None:
        logger.info(f"Adding {len(violations)} clique cuts...")
        for clique in violations:
            m.add_constr(model.clique_eqs.add(sum(model.x[i] for i in clique) <= 1))

    pm = PyomoModel(
        model,
        solver,
        cuts_separate=cuts_separate,
        cuts_enforce=cuts_enforce,
    )
    _pyomo_set_params(pm, params, solver)
    return pm


def _stab_read(data: Union[str, MaxWeightStableSetData]) -> MaxWeightStableSetData:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, MaxWeightStableSetData)
    return data


def _stab_separate(data: MaxWeightStableSetData, x_val: List[float]) -> List:
    # Check that we selected at most one vertex for each
    # clique in the graph (sum <= 1)
    violations: List[Any] = []
    for clique in nx.find_cliques(data.graph):
        if sum(x_val[i] for i in clique) > 1.0001:
            violations.append(sorted(clique))
    return violations
