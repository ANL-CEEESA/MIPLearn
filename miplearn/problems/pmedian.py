#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Optional, Union, Callable

import gurobipy as gp
import numpy as np
from gurobipy import quicksum, GRB
from scipy.spatial.distance import pdist, squareform
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel


@dataclass
class PMedianData:
    """Data for the capacitated p-median problem

    Args
    ----
    distances
        Matrix of distances between customer i and facility j.
    demands
        Customer demands.
    p
        Number of medians that need to be chosen.
    capacities
        Facility capacities.
    """

    distances: np.ndarray
    demands: np.ndarray
    p: int
    capacities: np.ndarray


class PMedianGenerator:
    """Random generator for the capacitated p-median problem.

    This class first decides the number of customers and the parameter `p` by
    sampling the provided `n` and `p` distributions, respectively. Then, for each
    customer `i`, the class builds its geographical location `(xi, yi)` by sampling
    the provided `x` and `y` distributions. For each `i`, the demand for customer `i`
    and the capacity of facility `i` are decided by sampling the distributions
    `demands` and `capacities`, respectively. Finally, the costs `w[i,j]` are set to
    the Euclidean distance between the locations of customers `i` and `j`.

    Parameters
    ----------
    x
        Probability distribution for the x-coordinate of the points.
    y
        Probability distribution for the y-coordinate of the points.
    n
        Probability distribution for the number of customer.
    p
        Probability distribution for the number of medians, or a callable that takes
        the number of customers and returns the number of medians (e.g., lambda n: n//10).
    demands
        Probability distribution for the customer demands.
    capacities
        Probability distribution for the facility capacities.
    """

    def __init__(
        self,
        x: rv_frozen = uniform(loc=0.0, scale=100.0),
        y: rv_frozen = uniform(loc=0.0, scale=100.0),
        n: rv_frozen = randint(low=100, high=101),
        p: Union[rv_frozen, Callable] = randint(low=10, high=11),
        demands: rv_frozen = uniform(loc=0, scale=20),
        capacities: rv_frozen = uniform(loc=0, scale=100),
    ):
        assert isinstance(x, rv_frozen), "x should be a SciPy probability distribution"
        assert isinstance(y, rv_frozen), "y should be a SciPy probability distribution"
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(p, rv_frozen) or callable(
            p
        ), "p should be a SciPy probability distribution or callable"
        assert isinstance(
            demands, rv_frozen
        ), "demands should be a SciPy probability distribution"
        assert isinstance(
            capacities, rv_frozen
        ), "capacities should be a SciPy probability distribution"

        self.x = x
        self.y = y
        self.n = n
        self.p = p
        self.demands = demands
        self.capacities = capacities

    def generate(self, n_samples: int) -> List[PMedianData]:
        def _sample() -> PMedianData:
            n = self.n.rvs()
            if callable(self.p):
                p = self.p(n)
            else:
                p = self.p.rvs()
            loc = np.array([(self.x.rvs(), self.y.rvs()) for _ in range(n)])
            distances = squareform(pdist(loc))
            demands = self.demands.rvs(n)
            capacities = self.capacities.rvs(n)

            data = PMedianData(
                distances=distances.round(2),
                demands=demands.round(2),
                p=p,
                capacities=capacities.round(2),
            )

            return data

        return [_sample() for _ in range(n_samples)]


class PMedianPerturber:
    """Perturbation generator for existing p-median instances.

    Takes an existing PMedianData instance and generates new instances by applying
    randomization factors to the existing distances, demands, and capacities while
    keeping the graph structure and parameter p fixed.
    """

    def __init__(
        self,
        distances_jitter: rv_frozen = uniform(loc=1.0, scale=0.0),
        demands_jitter: rv_frozen = uniform(loc=1.0, scale=0.0),
        capacities_jitter: rv_frozen = uniform(loc=1.0, scale=0.0),
    ):
        """Initialize the perturbation generator.

        Parameters
        ----------
        distances_jitter
            Probability distribution for randomization factors applied to distances.
        demands_jitter
            Probability distribution for randomization factors applied to demands.
        capacities_jitter
            Probability distribution for randomization factors applied to capacities.
        """
        self.distances_jitter = distances_jitter
        self.demands_jitter = demands_jitter
        self.capacities_jitter = capacities_jitter

    def perturb(
        self,
        instance: PMedianData,
        n_samples: int,
    ) -> List[PMedianData]:
        def _sample() -> PMedianData:
            n = instance.demands.shape[0]
            distances = instance.distances * self.distances_jitter.rvs(size=(n, n))
            distances = np.tril(distances) + np.triu(distances.T, 1)
            demands = instance.demands * self.demands_jitter.rvs(n)
            capacities = instance.capacities * self.capacities_jitter.rvs(n)

            return PMedianData(
                distances=distances.round(2),
                demands=demands.round(2),
                p=instance.p,
                capacities=capacities.round(2),
            )

        return [_sample() for _ in range(n_samples)]


def build_pmedian_model_gurobipy(data: Union[str, PMedianData]) -> GurobiModel:
    """Converts capacitated p-median data into a concrete Gurobipy model."""
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, PMedianData)

    model = gp.Model()
    n = len(data.demands)

    # Decision variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    # Objective function
    model.setObjective(
        quicksum(data.distances[i, j] * x[i, j] for i in range(n) for j in range(n))
    )

    # Eq: Must serve each customer
    model.addConstrs(
        (quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)),
        name="eq_demand",
    )

    # Eq: Must choose p medians
    model.addConstr(
        quicksum(y[j] for j in range(n)) == data.p,
        name="eq_choose",
    )

    # Eq: Must not exceed capacity
    model.addConstrs(
        (
            quicksum(data.demands[i] * x[i, j] for i in range(n))
            <= data.capacities[j] * y[j]
            for j in range(n)
        ),
        name="eq_capacity",
    )

    model.update()
    return GurobiModel(model)
