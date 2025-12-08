#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from dataclasses import dataclass
from typing import List, Optional, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel


@dataclass
class BinPackData:
    """Data for the bin packing problem.

    Parameters
    ----------
    sizes
        Sizes of the items
    capacity
        Capacity of the bin
    """

    sizes: np.ndarray
    capacity: int


class BinPackGenerator:
    """Random instance generator for the bin packing problem.

    Generates instances by sampling the user-provided probability distributions
    n, sizes and capacity to decide, respectively, the number of items, the sizes of
    the items and capacity of the bin. All values are sampled independently.

    Args
    ----
    n
        Probability distribution for the number of items.
    sizes
        Probability distribution for the item sizes.
    capacity
        Probability distribution for the bin capacity.
    """

    def __init__(
        self,
        n: rv_frozen,
        sizes: rv_frozen,
        capacity: rv_frozen,
    ) -> None:
        self.n = n
        self.sizes = sizes
        self.capacity = capacity

    def generate(self, n_samples: int) -> List[BinPackData]:
        """Generates random instances.

        Parameters
        ----------
        n_samples
            Number of samples to generate.
        """

        def _sample() -> BinPackData:
            n = self.n.rvs()
            sizes = self.sizes.rvs(n)
            capacity = self.capacity.rvs()
            return BinPackData(sizes.round(2), capacity.round(2))

        return [_sample() for _ in range(n_samples)]


class BinPackPerturber:
    """Perturbation generator for existing bin packing instances.

    Takes an existing BinPackData instance and generates new instances by perturbing
    its item sizes and bin capacity. The sizes of the items are set to `s_i * gamma_i`
    where `s_i` is the size of the i-th item in the reference instance and `gamma_i`
    is sampled from `sizes_jitter`. Similarly, the bin capacity is set to `B * beta`,
    where `B` is the reference bin capacity and `beta` is sampled from `capacity_jitter`.
    The number of items remains the same across all generated instances.

    Args
    ----
    sizes_jitter
        Probability distribution for the item size randomization.
    capacity_jitter
        Probability distribution for the bin capacity randomization.
    """

    def __init__(
        self,
        sizes_jitter: rv_frozen,
        capacity_jitter: rv_frozen,
    ) -> None:
        self.sizes_jitter = sizes_jitter
        self.capacity_jitter = capacity_jitter

    def perturb(
        self,
        instance: BinPackData,
        n_samples: int,
    ) -> List[BinPackData]:
        """Generates perturbed instances.

        Parameters
        ----------
        instance
            The reference instance to perturb.
        n_samples
            Number of samples to generate.
        """

        def _sample() -> BinPackData:
            n = instance.sizes.shape[0]
            sizes = instance.sizes * self.sizes_jitter.rvs(n)
            capacity = instance.capacity * self.capacity_jitter.rvs()
            return BinPackData(sizes.round(2), capacity.round(2))

        return [_sample() for _ in range(n_samples)]


def build_binpack_model_gurobipy(data: Union[str, BinPackData]) -> GurobiModel:
    """Converts bin packing problem data into a concrete Gurobipy model."""
    if isinstance(data, str):
        data = read_pkl_gz(data)
    assert isinstance(data, BinPackData)

    model = gp.Model()
    n = data.sizes.shape[0]

    # Var: Use bin
    y = model.addVars(n, name="y", vtype=GRB.BINARY)

    # Var: Assign item to bin
    x = model.addVars(n, n, name="x", vtype=GRB.BINARY)

    # Obj: Minimize number of bins
    model.setObjective(quicksum(y[i] for i in range(n)))

    # Eq: Enforce bin capacity
    model.addConstrs(
        (
            quicksum(data.sizes[i] * x[i, j] for i in range(n)) <= data.capacity * y[j]
            for j in range(n)
        ),
        name="eq_capacity",
    )

    # Eq: Must assign all items to bins
    model.addConstrs(
        (quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)),
        name="eq_assign",
    )

    model.update()
    return GurobiModel(model)
