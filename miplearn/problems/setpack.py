#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Union

import gurobipy as gp
import numpy as np
from gurobipy.gurobipy import GRB
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from .setcover import SetCoverGenerator
from miplearn.solvers.gurobi import GurobiModel
from ..io import read_pkl_gz


@dataclass
class SetPackData:
    costs: np.ndarray
    incidence_matrix: np.ndarray


class SetPackGenerator:
    def __init__(
        self,
        n_elements: rv_frozen = randint(low=50, high=51),
        n_sets: rv_frozen = randint(low=100, high=101),
        costs: rv_frozen = uniform(loc=0.0, scale=100.0),
        costs_jitter: rv_frozen = uniform(loc=-5.0, scale=10.0),
        K: rv_frozen = uniform(loc=25.0, scale=0.0),
        density: rv_frozen = uniform(loc=0.02, scale=0.00),
        fix_sets: bool = True,
    ) -> None:
        self.gen = SetCoverGenerator(
            n_elements=n_elements,
            n_sets=n_sets,
            costs=costs,
            costs_jitter=costs_jitter,
            K=K,
            density=density,
            fix_sets=fix_sets,
        )

    def generate(self, n_samples: int) -> List[SetPackData]:
        return [
            SetPackData(
                s.costs,
                s.incidence_matrix,
            )
            for s in self.gen.generate(n_samples)
        ]


def build_setpack_model(data: Union[str, SetPackData]) -> GurobiModel:
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
