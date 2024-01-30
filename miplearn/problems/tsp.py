#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Union

import gurobipy as gp
import networkx as nx
import numpy as np
import pyomo.environ as pe
from gurobipy import quicksum, GRB, tuplelist
from miplearn.io import read_pkl_gz
from miplearn.problems import _gurobipy_set_params, _pyomo_set_params
from miplearn.solvers.gurobi import GurobiModel
from scipy.spatial.distance import pdist, squareform
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.solvers.pyomo import PyomoModel

logger = logging.getLogger(__name__)


@dataclass
class TravelingSalesmanData:
    n_cities: int
    distances: np.ndarray


class TravelingSalesmanGenerator:
    """Random generator for the Traveling Salesman Problem."""

    def __init__(
        self,
        x: rv_frozen = uniform(loc=0.0, scale=1000.0),
        y: rv_frozen = uniform(loc=0.0, scale=1000.0),
        n: rv_frozen = randint(low=100, high=101),
        gamma: rv_frozen = uniform(loc=1.0, scale=0.0),
        fix_cities: bool = True,
        round: bool = True,
    ) -> None:
        """Initializes the problem generator.

        Initially, the generator creates n cities (x_1,y_1),...,(x_n,y_n) where n,
        x_i and y_i are sampled independently from the provided probability
        distributions `n`, `x` and `y`. For each (unordered) pair of cities (i,j),
        the distance d[i,j] between them is set to:

            d[i,j] = gamma[i,j] \\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}

        where gamma is sampled from the provided probability distribution `gamma`.

        If fix_cities=True, the list of cities is kept the same for all generated
        instances. The gamma values, and therefore also the distances, are still
        different.

        By default, all distances d[i,j] are rounded to the nearest integer.  If
        `round=False` is provided, this rounding will be disabled.

        Arguments
        ---------
        x: rv_continuous
            Probability distribution for the x-coordinate of each city.
        y: rv_continuous
            Probability distribution for the y-coordinate of each city.
        n: rv_discrete
            Probability distribution for the number of cities.
        fix_cities: bool
            If False, cities will be resampled for every generated instance. Otherwise, list
            of cities will be computed once, during the constructor.
        round: bool
            If True, distances are rounded to the nearest integer.
        """
        assert isinstance(x, rv_frozen), "x should be a SciPy probability distribution"
        assert isinstance(y, rv_frozen), "y should be a SciPy probability distribution"
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(
            gamma,
            rv_frozen,
        ), "gamma should be a SciPy probability distribution"
        self.x = x
        self.y = y
        self.n = n
        self.gamma = gamma
        self.round = round

        if fix_cities:
            self.fixed_n: Optional[int]
            self.fixed_cities: Optional[np.ndarray]
            self.fixed_n, self.fixed_cities = self._generate_cities()
        else:
            self.fixed_n = None
            self.fixed_cities = None

    def generate(self, n_samples: int) -> List[TravelingSalesmanData]:
        def _sample() -> TravelingSalesmanData:
            if self.fixed_cities is not None:
                assert self.fixed_n is not None
                n, cities = self.fixed_n, self.fixed_cities
            else:
                n, cities = self._generate_cities()
            distances = squareform(pdist(cities)) * self.gamma.rvs(size=(n, n))
            distances = np.tril(distances) + np.triu(distances.T, 1)
            if self.round:
                distances = distances.round()
            return TravelingSalesmanData(n, distances)

        return [_sample() for _ in range(n_samples)]

    def _generate_cities(self) -> Tuple[int, np.ndarray]:
        n = self.n.rvs()
        cities = np.array([(self.x.rvs(), self.y.rvs()) for _ in range(n)])
        return n, cities


def build_tsp_model_gurobipy(
    data: Union[str, TravelingSalesmanData],
    params: Optional[dict[str, Any]] = None,
) -> GurobiModel:

    model = gp.Model()
    _gurobipy_set_params(model, params)

    data = _tsp_read(data)
    edges = tuplelist(
        (i, j) for i in range(data.n_cities) for j in range(i + 1, data.n_cities)
    )

    # Decision variables
    x = model.addVars(edges, vtype=GRB.BINARY, name="x")

    model._x = x
    model._edges = edges
    model._n_cities = data.n_cities

    # Objective function
    model.setObjective(quicksum(x[(i, j)] * data.distances[i, j] for (i, j) in edges))

    # Eq: Must choose two edges adjacent to each node
    model.addConstrs(
        (
            quicksum(x[min(i, j), max(i, j)] for j in range(data.n_cities) if i != j)
            == 2
            for i in range(data.n_cities)
        ),
        name="eq_degree",
    )

    def lazy_separate(model: GurobiModel) -> List[Any]:
        violations = []
        x = model.inner.cbGetSolution(model.inner._x)
        selected_edges = [e for e in model.inner._edges if x[e] > 0.5]
        graph = nx.Graph()
        graph.add_edges_from(selected_edges)
        for component in list(nx.connected_components(graph)):
            if len(component) < model.inner._n_cities:
                cut_edges = tuple(
                    (e[0], e[1])
                    for e in model.inner._edges
                    if (e[0] in component and e[1] not in component)
                    or (e[0] not in component and e[1] in component)
                )
                violations.append(cut_edges)
        return violations

    def lazy_enforce(model: GurobiModel, violations: List[Any]) -> None:
        for violation in violations:
            model.add_constr(
                quicksum(model.inner._x[e[0], e[1]] for e in violation) >= 2
            )
        logger.info(f"tsp: added {len(violations)} subtour elimination constraints")

    model.update()

    return GurobiModel(
        model,
        lazy_separate=lazy_separate,
        lazy_enforce=lazy_enforce,
    )


def build_tsp_model_pyomo(
    data: Union[str, TravelingSalesmanData],
    solver: str = "gurobi_persistent",
    params: Optional[dict[str, Any]] = None,
) -> PyomoModel:

    model = pe.ConcreteModel()
    data = _tsp_read(data)

    edges = tuplelist(
        (i, j) for i in range(data.n_cities) for j in range(i + 1, data.n_cities)
    )

    # Decision variables
    model.x = pe.Var(edges, domain=pe.Boolean, name="x")
    model.obj = pe.Objective(
        expr=sum(model.x[i, j] * data.distances[i, j] for (i, j) in edges)
    )

    # Eq: Must choose two edges adjacent to each node
    model.degree_eqs = pe.ConstraintList()
    for i in range(data.n_cities):
        model.degree_eqs.add(
            sum(model.x[min(i, j), max(i, j)] for j in range(data.n_cities) if i != j)
            == 2
        )

    # Eq: Subtour elimination
    model.subtour_eqs = pe.ConstraintList()

    def lazy_separate(m: PyomoModel) -> List[Any]:
        violations = []
        m.solver.cbGetSolution([model.x[e] for e in edges])
        x_val = {e: model.x[e].value for e in edges}
        selected_edges = [e for e in edges if x_val[e] > 0.5]
        graph = nx.Graph()
        graph.add_edges_from(selected_edges)
        for component in list(nx.connected_components(graph)):
            if len(component) < data.n_cities:
                cut_edges = tuple(
                    (e[0], e[1])
                    for e in edges
                    if (e[0] in component and e[1] not in component)
                    or (e[0] not in component and e[1] in component)
                )
                violations.append(cut_edges)
        return violations

    def lazy_enforce(m: PyomoModel, violations: List[Any]) -> None:
        logger.warning(f"Adding {len(violations)} subtour elimination constraints...")
        for violation in violations:
            m.add_constr(
                model.subtour_eqs.add(sum(model.x[e[0], e[1]] for e in violation) >= 2)
            )

    pm = PyomoModel(
        model,
        solver,
        lazy_separate=lazy_separate,
        lazy_enforce=lazy_enforce,
    )
    _pyomo_set_params(pm, params, solver)
    return pm


def _tsp_read(data: Union[str, TravelingSalesmanData]) -> TravelingSalesmanData:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, TravelingSalesmanData)
    return data
