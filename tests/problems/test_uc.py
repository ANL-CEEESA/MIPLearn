#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from scipy.stats import uniform, randint

from miplearn.problems.uc import (
    UnitCommitmentData,
    build_uc_model_gurobipy,
    UnitCommitmentGenerator,
)


def test_generator() -> None:
    np.random.seed(42)
    gen = UnitCommitmentGenerator(
        n_units=randint(low=3, high=4),
        n_periods=randint(low=4, high=5),
        max_power=uniform(loc=50, scale=450),
        min_power=uniform(loc=0.25, scale=0.5),
        cost_startup=uniform(loc=1, scale=1),
        cost_prod=uniform(loc=1, scale=1),
        cost_fixed=uniform(loc=1, scale=1),
        min_uptime=randint(low=1, high=8),
        min_downtime=randint(low=1, high=8),
        cost_jitter=uniform(loc=0.75, scale=0.5),
        demand_jitter=uniform(loc=0.9, scale=0.2),
        fix_units=True,
    )
    data = gen.generate(2)

    assert data[0].demand.tolist() == [430.3, 518.65, 448.16, 860.61]
    assert data[0].min_power.tolist() == [120.05, 156.73, 124.44]
    assert data[0].max_power.tolist() == [218.54, 477.82, 379.4]
    assert data[0].min_uptime.tolist() == [3, 3, 5]
    assert data[0].min_downtime.tolist() == [4, 3, 6]
    assert data[0].cost_startup.tolist() == [1.06, 1.72, 1.94]
    assert data[0].cost_prod.tolist() == [1.0, 1.99, 1.62]
    assert data[0].cost_fixed.tolist() == [1.61, 1.01, 1.02]

    assert data[1].demand.tolist() == [407.3, 476.18, 458.77, 840.38]
    assert data[1].min_power.tolist() == [120.05, 156.73, 124.44]
    assert data[1].max_power.tolist() == [218.54, 477.82, 379.4]
    assert data[1].min_uptime.tolist() == [3, 3, 5]
    assert data[1].min_downtime.tolist() == [4, 3, 6]
    assert data[1].cost_startup.tolist() == [1.32, 1.69, 2.29]
    assert data[1].cost_prod.tolist() == [1.09, 1.94, 1.23]
    assert data[1].cost_fixed.tolist() == [1.97, 1.04, 0.96]


def test_uc() -> None:
    data = UnitCommitmentData(
        demand=np.array([10, 12, 15, 10, 8, 5]),
        min_power=np.array([5, 5, 10]),
        max_power=np.array([10, 8, 20]),
        min_uptime=np.array([4, 3, 2]),
        min_downtime=np.array([4, 3, 2]),
        cost_startup=np.array([100, 120, 200]),
        cost_prod=np.array([1.0, 1.25, 1.5]),
        cost_fixed=np.array([10, 12, 9]),
    )
    model = build_uc_model_gurobipy(data)
    model.optimize()
    assert model.inner.objVal == 154.5


if __name__ == "__main__":
    data = UnitCommitmentGenerator().generate(1)[0]
    model = build_uc_model_gurobipy(data)
    model.optimize()
