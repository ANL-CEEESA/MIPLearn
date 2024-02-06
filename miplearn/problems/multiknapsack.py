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

    Instances have a random number of items (or variables) and a random number of
    knapsacks (or constraints), as specified by the provided probability
    distributions `n` and `m`, respectively. The weight of each item `i` on knapsack
    `j` is sampled independently from the provided distribution `w`. The capacity of
    knapsack `j` is set to ``alpha_j * sum(w[i,j] for i in range(n))``,
    where `alpha_j`, the tightness ratio, is sampled from the provided probability
    distribution `alpha`.

    To make the instances more challenging, the costs of the items are linearly
    correlated to their average weights. More specifically, the weight of each item
    `i` is set to ``sum(w[i,j]/m for j in range(m)) + K * u_i``, where `K`,
    the correlation coefficient, and `u_i`, the correlation multiplier, are sampled
    from the provided probability distributions. Note that `K` is only sample once
    for the entire instance.

    If `fix_w=True`, then `weights[i,j]` are kept the same in all generated
    instances. This also implies that n and m are kept fixed. Although the prices and
    capacities are derived from `weights[i,j]`, as long as `u` and `K` are not
    constants, the generated instances will still not be completely identical.

    If a probability distribution `w_jitter` is provided, then item weights will be
    set to ``w[i,j] * gamma[i,j]`` where `gamma[i,j]` is sampled from `w_jitter`.
    When combined with `fix_w=True`, this argument may be used to generate instances
    where the weight of each item is roughly the same, but not exactly identical,
    across all instances. The prices of the items and the capacities of the knapsacks
    will be calculated as above, but using these perturbed weights instead.

    By default, all generated prices, weights and capacities are rounded to the
    nearest integer number. If `round=False` is provided, this rounding will be
    disabled.

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
    fix_w: boolean
        If true, weights are kept the same (minus the noise from w_jitter) in all
        instances.
    w_jitter: rv_continuous
        Probability distribution for random noise added to the weights.
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
        fix_w: bool = False,
        w_jitter: rv_frozen = uniform(loc=1.0, scale=0.0),
        p_jitter: rv_frozen = uniform(loc=1.0, scale=0.0),
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
        assert isinstance(fix_w, bool), "fix_w should be boolean"
        assert isinstance(
            w_jitter, rv_frozen
        ), "w_jitter should be a SciPy probability distribution"

        self.n = n
        self.m = m
        self.w = w
        self.u = u
        self.K = K
        self.alpha = alpha
        self.w_jitter = w_jitter
        self.p_jitter = p_jitter
        self.round = round
        self.fix_n: Optional[int] = None
        self.fix_m: Optional[int] = None
        self.fix_w: Optional[np.ndarray] = None
        self.fix_u: Optional[np.ndarray] = None
        self.fix_K: Optional[float] = None

        if fix_w:
            self.fix_n = self.n.rvs()
            self.fix_m = self.m.rvs()
            self.fix_w = np.array([self.w.rvs(self.fix_n) for _ in range(self.fix_m)])
            self.fix_u = self.u.rvs(self.fix_n)
            self.fix_K = self.K.rvs()

    def generate(self, n_samples: int) -> List[MultiKnapsackData]:
        def _sample() -> MultiKnapsackData:
            if self.fix_w is not None:
                assert self.fix_m is not None
                assert self.fix_n is not None
                assert self.fix_u is not None
                assert self.fix_K is not None
                n = self.fix_n
                m = self.fix_m
                w = self.fix_w
                u = self.fix_u
                K = self.fix_K
            else:
                n = self.n.rvs()
                m = self.m.rvs()
                w = np.array([self.w.rvs(n) for _ in range(m)])
                u = self.u.rvs(n)
                K = self.K.rvs()
            w = w * np.array([self.w_jitter.rvs(n) for _ in range(m)])
            alpha = self.alpha.rvs(m)
            p = np.array(
                [w[:, j].sum() / m + K * u[j] for j in range(n)]
            ) * self.p_jitter.rvs(n)
            b = np.array([w[i, :].sum() * alpha[i] for i in range(m)])
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
