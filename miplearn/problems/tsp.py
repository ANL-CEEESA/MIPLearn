#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict

import networkx as nx
import numpy as np
import pyomo.environ as pe
from overrides import overrides
from scipy.spatial.distance import pdist, squareform
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.instance.base import Instance
from miplearn.solvers.learning import InternalSolver
from miplearn.solvers.pyomo.base import BasePyomoSolver
from miplearn.types import ConstraintName


@dataclass
class TravelingSalesmanData:
    n_cities: int
    distances: np.ndarray


class TravelingSalesmanInstance(Instance):
    """An instance ot the Traveling Salesman Problem.

    Given a list of cities and the distance between each pair of cities, the problem
    asks for the shortest route starting at the first city, visiting each other city
    exactly once, then returning to the first city. This problem is a generalization
    of the Hamiltonian path problem, one of Karp's 21 NP-complete problems.
    """

    def __init__(self, n_cities: int, distances: np.ndarray) -> None:
        super().__init__()
        assert isinstance(distances, np.ndarray)
        assert distances.shape == (n_cities, n_cities)
        self.n_cities = n_cities
        self.distances = distances
        self.edges = [
            (i, j) for i in range(self.n_cities) for j in range(i + 1, self.n_cities)
        ]

    @overrides
    def to_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        model.x = pe.Var(self.edges, domain=pe.Binary)
        model.obj = pe.Objective(
            expr=sum(model.x[i, j] * self.distances[i, j] for (i, j) in self.edges),
            sense=pe.minimize,
        )
        model.eq_degree = pe.ConstraintList()
        model.eq_subtour = pe.ConstraintList()
        for i in range(self.n_cities):
            model.eq_degree.add(
                sum(
                    model.x[min(i, j), max(i, j)]
                    for j in range(self.n_cities)
                    if i != j
                )
                == 2
            )
        return model

    @overrides
    def find_violated_lazy_constraints(
        self,
        solver: InternalSolver,
        model: Any,
    ) -> Dict[ConstraintName, List]:
        selected_edges = [e for e in self.edges if model.x[e].value > 0.5]
        graph = nx.Graph()
        graph.add_edges_from(selected_edges)
        violations = {}
        for c in list(nx.connected_components(graph)):
            if len(c) < self.n_cities:
                cname = ("st[" + ",".join(map(str, c)) + "]").encode()
                violations[cname] = list(c)
        return violations

    @overrides
    def enforce_lazy_constraint(
        self,
        solver: InternalSolver,
        model: Any,
        component: List,
    ) -> None:
        assert isinstance(solver, BasePyomoSolver)
        cut_edges = [
            e
            for e in self.edges
            if (e[0] in component and e[1] not in component)
            or (e[0] not in component and e[1] in component)
        ]
        constr = model.eq_subtour.add(expr=sum(model.x[e] for e in cut_edges) >= 2)
        solver.add_constraint(constr)


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

            d[i,j] = gamma[i,j] \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}

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
