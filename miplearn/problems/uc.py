#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from math import pi
from typing import List, Optional, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel


@dataclass
class UnitCommitmentData:
    demand: np.ndarray
    min_power: np.ndarray
    max_power: np.ndarray
    min_uptime: np.ndarray
    min_downtime: np.ndarray
    cost_startup: np.ndarray
    cost_prod: np.ndarray
    cost_fixed: np.ndarray


class UnitCommitmentGenerator:
    def __init__(
        self,
        n_units: rv_frozen = randint(low=1_000, high=1_001),
        n_periods: rv_frozen = randint(low=72, high=73),
        max_power: rv_frozen = uniform(loc=50, scale=450),
        min_power: rv_frozen = uniform(loc=0.5, scale=0.25),
        cost_startup: rv_frozen = uniform(loc=0, scale=10_000),
        cost_prod: rv_frozen = uniform(loc=0, scale=50),
        cost_fixed: rv_frozen = uniform(loc=0, scale=1_000),
        min_uptime: rv_frozen = randint(low=2, high=8),
        min_downtime: rv_frozen = randint(low=2, high=8),
        cost_jitter: rv_frozen = uniform(loc=0.75, scale=0.5),
        demand_jitter: rv_frozen = uniform(loc=0.9, scale=0.2),
        fix_units: bool = False,
    ) -> None:
        self.n_units = n_units
        self.n_periods = n_periods
        self.max_power = max_power
        self.min_power = min_power
        self.cost_startup = cost_startup
        self.cost_prod = cost_prod
        self.cost_fixed = cost_fixed
        self.min_uptime = min_uptime
        self.min_downtime = min_downtime
        self.cost_jitter = cost_jitter
        self.demand_jitter = demand_jitter
        self.fix_units = fix_units
        self.ref_data: Optional[UnitCommitmentData] = None

    def generate(self, n_samples: int) -> List[UnitCommitmentData]:
        def _sample() -> UnitCommitmentData:
            if self.ref_data is None:
                T = self.n_periods.rvs()
                G = self.n_units.rvs()

                # Generate unit parameteres
                max_power = self.max_power.rvs(G)
                min_power = max_power * self.min_power.rvs(G)
                max_power = max_power
                min_uptime = self.min_uptime.rvs(G)
                min_downtime = self.min_downtime.rvs(G)
                cost_startup = self.cost_startup.rvs(G)
                cost_prod = self.cost_prod.rvs(G)
                cost_fixed = self.cost_fixed.rvs(G)
                capacity = max_power.sum()

                # Generate periodic demand in the range [0.4, 0.8] * capacity, with a peak every 12 hours.
                demand = np.sin([i / 6 * pi for i in range(T)])
                demand *= uniform(loc=0, scale=1).rvs(T)
                demand -= demand.min()
                demand /= demand.max() / 0.4
                demand += 0.4
                demand *= capacity
            else:
                T, G = len(self.ref_data.demand), len(self.ref_data.max_power)
                demand = self.ref_data.demand * self.demand_jitter.rvs(T)
                min_power = self.ref_data.min_power
                max_power = self.ref_data.max_power
                min_uptime = self.ref_data.min_uptime
                min_downtime = self.ref_data.min_downtime
                cost_startup = self.ref_data.cost_startup * self.cost_jitter.rvs(G)
                cost_prod = self.ref_data.cost_prod * self.cost_jitter.rvs(G)
                cost_fixed = self.ref_data.cost_fixed * self.cost_jitter.rvs(G)

            data = UnitCommitmentData(
                demand.round(2),
                min_power.round(2),
                max_power.round(2),
                min_uptime,
                min_downtime,
                cost_startup.round(2),
                cost_prod.round(2),
                cost_fixed.round(2),
            )

            if self.ref_data is None and self.fix_units:
                self.ref_data = data

            return data

        return [_sample() for _ in range(n_samples)]


def build_uc_model_gurobipy(data: Union[str, UnitCommitmentData]) -> GurobiModel:
    """
    Models the unit commitment problem according to equations (1)-(5) of:

        Bendotti, P., Fouilhoux, P. & Rottner, C. The min-up/min-down unit
        commitment polytope. J Comb Optim 36, 1024-1058 (2018).
        https://doi.org/10.1007/s10878-018-0273-y

    """
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, UnitCommitmentData)

    T = len(data.demand)
    G = len(data.min_power)
    D = data.demand
    Pmin, Pmax = data.min_power, data.max_power
    L = data.min_uptime
    l = data.min_downtime

    model = gp.Model()
    is_on = model.addVars(G, T, vtype=GRB.BINARY, name="is_on")
    switch_on = model.addVars(G, T, vtype=GRB.BINARY, name="switch_on")
    prod = model.addVars(G, T, name="prod")

    # Objective function
    model.setObjective(
        quicksum(
            is_on[g, t] * data.cost_fixed[g]
            + switch_on[g, t] * data.cost_startup[g]
            + prod[g, t] * data.cost_prod[g]
            for g in range(G)
            for t in range(T)
        )
    )

    # Eq 1: Minimum up-time constraint: If unit g is down at time t, then it
    # cannot have start up during the previous L[g] periods.
    model.addConstrs(
        (
            quicksum(switch_on[g, k] for k in range(t - L[g] + 1, t + 1)) <= is_on[g, t]
            for g in range(G)
            for t in range(L[g] - 1, T)
        ),
        name="eq_min_uptime",
    )

    # Eq 2: Minimum down-time constraint: Symmetric to the minimum-up constraint.
    model.addConstrs(
        (
            quicksum(switch_on[g, k] for k in range(t - l[g] + 1, t + 1))
            <= 1 - is_on[g, t - l[g] + 1]
            for g in range(G)
            for t in range(l[g] - 1, T)
        ),
        name="eq_min_downtime",
    )

    # Eq 3: Ensures that if unit g start up at time t, then the start-up variable
    # must be one.
    model.addConstrs(
        (
            switch_on[g, t] >= is_on[g, t] - is_on[g, t - 1]
            for g in range(G)
            for t in range(1, T)
        ),
        name="eq_startup",
    )

    # Eq 4: Ensures that demand is satisfied at each time period.
    model.addConstrs(
        (quicksum(prod[g, t] for g in range(G)) >= D[t] for t in range(T)),
        name="eq_demand",
    )

    # Eq 5: Sets the bounds to the quantity of power produced by each unit.
    model.addConstrs(
        (Pmin[g] * is_on[g, t] <= prod[g, t] for g in range(G) for t in range(T)),
        name="eq_prod_lb",
    )
    model.addConstrs(
        (prod[g, t] <= Pmax[g] * is_on[g, t] for g in range(G) for t in range(T)),
        name="eq_prod_ub",
    )
    model.update()

    return GurobiModel(model)
