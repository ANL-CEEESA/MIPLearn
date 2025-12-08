#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Union, Callable

import gurobipy as gp
import numpy as np
import pyomo.environ as pe
from gurobipy import GRB
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel
from miplearn.solvers.pyomo import PyomoModel


@dataclass
class SetCoverData:
    costs: np.ndarray
    incidence_matrix: np.ndarray


class SetCoverGenerator:
    """Random instance generator for the Set Cover Problem.

    Generates instances by creating a new random incidence matrix for each
    instance, where the number of elements, sets, density, and costs are sampled
    from user-provided probability distributions.
    """

    def __init__(
        self,
        n_elements: rv_frozen = randint(low=50, high=51),
        n_sets: Union[rv_frozen, Callable] = randint(low=100, high=101),
        costs: rv_frozen = uniform(loc=0.0, scale=100.0),
        K: rv_frozen = uniform(loc=25.0, scale=0.0),
        density: rv_frozen = uniform(loc=0.02, scale=0.00),
    ):
        """Initialize the problem generator.

        Parameters
        ----------
        n_elements: rv_discrete
            Probability distribution for number of elements.
        n_sets: rv_discrete or callable
            Probability distribution for number of sets, or a callable that takes
            the number of elements and returns the number of sets.
        costs: rv_continuous
            Probability distribution for base set costs.
        K: rv_continuous
            Probability distribution for cost scaling factor based on set size.
        density: rv_continuous
            Probability distribution for incidence matrix density.
        """
        assert isinstance(
            n_elements, rv_frozen
        ), "n_elements should be a SciPy probability distribution"
        assert isinstance(n_sets, rv_frozen) or callable(
            n_sets
        ), "n_sets should be a SciPy probability distribution or callable"
        assert isinstance(
            costs, rv_frozen
        ), "costs should be a SciPy probability distribution"
        assert isinstance(K, rv_frozen), "K should be a SciPy probability distribution"
        assert isinstance(
            density, rv_frozen
        ), "density should be a SciPy probability distribution"
        self.n_elements = n_elements
        self.n_sets = n_sets
        self.costs = costs
        self.density = density
        self.K = K

    def generate(self, n_samples: int) -> List[SetCoverData]:
        def _sample() -> SetCoverData:
            n_elements = self.n_elements.rvs()
            if callable(self.n_sets):
                n_sets = self.n_sets(n_elements)
            else:
                n_sets = self.n_sets.rvs()
            density = self.density.rvs()

            incidence_matrix = np.random.rand(n_elements, n_sets) < density
            incidence_matrix = incidence_matrix.astype(int)

            # Ensure each element belongs to at least one set
            for j in range(n_elements):
                if incidence_matrix[j, :].sum() == 0:
                    incidence_matrix[j, randint(low=0, high=n_sets).rvs()] = 1

            # Ensure each set contains at least one element
            for i in range(n_sets):
                if incidence_matrix[:, i].sum() == 0:
                    incidence_matrix[randint(low=0, high=n_elements).rvs(), i] = 1

            costs = self.costs.rvs(n_sets) + self.K.rvs() * incidence_matrix.sum(axis=0)
            return SetCoverData(
                costs=costs.round(2),
                incidence_matrix=incidence_matrix,
            )

        return [_sample() for _ in range(n_samples)]


class SetCoverPerturber:
    """Perturbation generator for existing Set Cover instances.

    Takes an existing SetCoverData instance and generates new instances
    by applying randomization factors to the existing costs while keeping the
    incidence matrix fixed.
    """

    def __init__(
        self,
        costs_jitter: rv_frozen = uniform(loc=0.9, scale=0.2),
    ):
        """Initialize the perturbation generator.

        Parameters
        ----------
        costs_jitter: rv_continuous
            Probability distribution for randomization factors applied to set costs.
        """
        assert isinstance(
            costs_jitter, rv_frozen
        ), "costs_jitter should be a SciPy probability distribution"
        self.costs_jitter = costs_jitter

    def perturb(
        self,
        instance: SetCoverData,
        n_samples: int,
    ) -> List[SetCoverData]:
        def _sample() -> SetCoverData:
            (_, n_sets) = instance.incidence_matrix.shape
            jitter_factors = self.costs_jitter.rvs(n_sets)
            costs = np.round(instance.costs * jitter_factors, 2)
            return SetCoverData(
                costs=costs,
                incidence_matrix=instance.incidence_matrix,
            )

        return [_sample() for _ in range(n_samples)]


def build_setcover_model_gurobipy(data: Union[str, SetCoverData]) -> GurobiModel:
    data = _read_setcover_data(data)
    (n_elements, n_sets) = data.incidence_matrix.shape
    model = gp.Model()
    x = model.addMVar(n_sets, vtype=GRB.BINARY, name="x")
    model.addConstr(data.incidence_matrix @ x >= np.ones(n_elements), name="eqs")
    model.setObjective(data.costs @ x)
    model.update()
    return GurobiModel(model)


def build_setcover_model_pyomo(
    data: Union[str, SetCoverData],
    solver: str = "gurobi_persistent",
) -> PyomoModel:
    data = _read_setcover_data(data)
    (n_elements, n_sets) = data.incidence_matrix.shape
    model = pe.ConcreteModel()
    model.sets = pe.Set(initialize=range(n_sets))
    model.x = pe.Var(model.sets, domain=pe.Boolean, name="x")
    model.eqs = pe.Constraint(pe.Any)
    for i in range(n_elements):
        model.eqs[i] = (
            sum(data.incidence_matrix[i, j] * model.x[j] for j in range(n_sets)) >= 1
        )
    model.obj = pe.Objective(
        expr=sum(data.costs[j] * model.x[j] for j in range(n_sets))
    )
    return PyomoModel(model, solver)


def _read_setcover_data(data: Union[str, SetCoverData]) -> SetCoverData:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, SetCoverData)
    return data
