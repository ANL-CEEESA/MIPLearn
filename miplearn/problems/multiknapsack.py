#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Optional, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel


@dataclass
class MultiKnapsackData:
    """Data for the multi-dimensional knapsack problem

    Args
    ----
    prices
        Item prices.
    capacities
        Knapsack capacities.
    weights
        Matrix of item weights.
    """

    prices: np.ndarray
    capacities: np.ndarray
    weights: np.ndarray


# noinspection PyPep8Naming
class MultiKnapsackGenerator:
    """Random instance generator for the multi-dimensional knapsack problem.

    Generates new instances by creating random items and knapsacks according to the
    provided probability distributions. Each instance has a random number of items
    (variables) and knapsacks (constraints), with weights, prices, and capacities
    sampled independently.

    Parameters
    ----------
    n: rv_discrete
        Probability distribution for the number of items (or variables).
    m: rv_discrete
        Probability distribution for the number of knapsacks (or constraints).
    w: rv_continuous
        Probability distribution for the item weights.
    K: rv_continuous
        Probability distribution for the profit correlation coefficient.
    u: rv_continuous
        Probability distribution for the profit multiplier.
    alpha: rv_continuous
        Probability distribution for the tightness ratio.
    round: boolean
        If true, all prices, weights and capacities are rounded to the nearest
        integer.
    """

    def __init__(
        self,
        n: rv_frozen = randint(low=100, high=101),
        m: rv_frozen = randint(low=30, high=31),
        w: rv_frozen = randint(low=0, high=1000),
        K: rv_frozen = randint(low=500, high=501),
        u: rv_frozen = uniform(loc=0.0, scale=1.0),
        alpha: rv_frozen = uniform(loc=0.25, scale=0.0),
        round: bool = True,
    ):
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(m, rv_frozen), "m should be a SciPy probability distribution"
        assert isinstance(w, rv_frozen), "w should be a SciPy probability distribution"
        assert isinstance(K, rv_frozen), "K should be a SciPy probability distribution"
        assert isinstance(u, rv_frozen), "u should be a SciPy probability distribution"
        assert isinstance(
            alpha, rv_frozen
        ), "alpha should be a SciPy probability distribution"

        self.n = n
        self.m = m
        self.w = w
        self.u = u
        self.K = K
        self.alpha = alpha
        self.round = round

    def generate(self, n_samples: int) -> List[MultiKnapsackData]:
        def _sample() -> MultiKnapsackData:
            n = self.n.rvs()
            m = self.m.rvs()
            w = np.array([self.w.rvs(n) for _ in range(m)])
            u = self.u.rvs(n)
            K = self.K.rvs()
            alpha = self.alpha.rvs(m)
            p = np.array([w[:, j].sum() / m + K * u[j] for j in range(n)])
            b = np.array([w[i, :].sum() * alpha[i] for i in range(m)])
            if self.round:
                p = p.round()
                b = b.round()
                w = w.round()
            return MultiKnapsackData(p, b, w)

        return [_sample() for _ in range(n_samples)]


class MultiKnapsackPerturber:
    """Perturbation generator for existing multi-dimensional knapsack instances.

    Takes an existing MultiKnapsackData instance and generates new instances by
    applying randomization factors to the existing weights and prices while keeping
    the structure (number of items and knapsacks) fixed.

    Parameters
    ----------
    w_jitter: rv_continuous
        Probability distribution for randomization factors applied to item weights.
    p_jitter: rv_continuous
        Probability distribution for randomization factors applied to item prices.
    alpha_jitter: rv_continuous
        Probability distribution for randomization factors applied to knapsack capacities.
    round: boolean
        If true, all perturbed prices, weights and capacities are rounded to the
        nearest integer.
    """

    def __init__(
        self,
        w_jitter: rv_frozen = uniform(loc=0.9, scale=0.2),
        p_jitter: rv_frozen = uniform(loc=0.9, scale=0.2),
        alpha_jitter: rv_frozen = uniform(loc=0.9, scale=0.2),
        round: bool = True,
    ):
        assert isinstance(
            w_jitter, rv_frozen
        ), "w_jitter should be a SciPy probability distribution"
        assert isinstance(
            p_jitter, rv_frozen
        ), "p_jitter should be a SciPy probability distribution"
        assert isinstance(
            alpha_jitter, rv_frozen
        ), "alpha_jitter should be a SciPy probability distribution"

        self.w_jitter = w_jitter
        self.p_jitter = p_jitter
        self.alpha_jitter = alpha_jitter
        self.round = round

    def perturb(
        self,
        instance: MultiKnapsackData,
        n_samples: int,
    ) -> List[MultiKnapsackData]:
        def _sample() -> MultiKnapsackData:
            m, n = instance.weights.shape
            w_factors = np.array([self.w_jitter.rvs(n) for _ in range(m)])
            p_factors = self.p_jitter.rvs(n)
            alpha_factors = self.alpha_jitter.rvs(m)

            w = instance.weights * w_factors
            p = instance.prices * p_factors
            b = instance.capacities * alpha_factors

            if self.round:
                p = p.round()
                b = b.round()
                w = w.round()
            return MultiKnapsackData(p, b, w)

        return [_sample() for _ in range(n_samples)]


def build_multiknapsack_model_gurobipy(
    data: Union[str, MultiKnapsackData]
) -> GurobiModel:
    """Converts multi-knapsack problem data into a concrete Gurobipy model."""
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, MultiKnapsackData)

    model = gp.Model()
    m, n = data.weights.shape
    x = model.addMVar(n, vtype=GRB.BINARY, name="x")
    model.addConstr(data.weights @ x <= data.capacities)
    model.setObjective(-data.prices @ x)
    model.update()
    return GurobiModel(model)
