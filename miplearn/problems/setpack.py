#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from .setcover import SetCoverGenerator, SetCoverPerturber
from miplearn.solvers.gurobi import GurobiModel
from ..io import read_pkl_gz


@dataclass
class SetPackData:
    costs: np.ndarray
    incidence_matrix: np.ndarray


class SetPackGenerator:
    """Random instance generator for the Set Packing Problem.

    Generates instances by creating a new random incidence matrix for each
    instance, where the number of elements, sets, density, and costs are sampled
    from user-provided probability distributions.
    """

    def __init__(
        self,
        n_elements: rv_frozen = randint(low=50, high=51),
        n_sets: rv_frozen = randint(low=100, high=101),
        costs: rv_frozen = uniform(loc=0.0, scale=100.0),
        K: rv_frozen = uniform(loc=25.0, scale=0.0),
        density: rv_frozen = uniform(loc=0.02, scale=0.00),
    ) -> None:
        """Initialize the problem generator.

        Parameters
        ----------
        n_elements: rv_discrete
            Probability distribution for number of elements.
        n_sets: rv_discrete
            Probability distribution for number of sets.
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
        assert isinstance(
            n_sets, rv_frozen
        ), "n_sets should be a SciPy probability distribution"
        assert isinstance(
            costs, rv_frozen
        ), "costs should be a SciPy probability distribution"
        assert isinstance(K, rv_frozen), "K should be a SciPy probability distribution"
        assert isinstance(
            density, rv_frozen
        ), "density should be a SciPy probability distribution"
        self.gen = SetCoverGenerator(
            n_elements=n_elements,
            n_sets=n_sets,
            costs=costs,
            K=K,
            density=density,
        )

    def generate(self, n_samples: int) -> List[SetPackData]:
        return [
            SetPackData(
                s.costs,
                s.incidence_matrix,
            )
            for s in self.gen.generate(n_samples)
        ]


class SetPackPerturber:
    """Perturbation generator for existing Set Packing instances.

    Takes an existing SetPackData instance and generates new instances
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
        instance: SetPackData,
        n_samples: int,
    ) -> List[SetPackData]:
        def _sample() -> SetPackData:
            (_, n_sets) = instance.incidence_matrix.shape
            jitter_factors = self.costs_jitter.rvs(n_sets)
            costs = np.round(instance.costs * jitter_factors, 2)
            return SetPackData(
                costs=costs,
                incidence_matrix=instance.incidence_matrix,
            )

        return [_sample() for _ in range(n_samples)]


def build_setpack_model_gurobipy(data: Union[str, SetPackData]) -> GurobiModel:
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, SetPackData)
    (n_elements, n_sets) = data.incidence_matrix.shape
    model = gp.Model()
    x = model.addMVar(n_sets, vtype=GRB.BINARY, name="x")
    model.addConstr(data.incidence_matrix @ x <= np.ones(n_elements))
    model.setObjective(-data.costs @ x)
    model.update()
    return GurobiModel(model)
