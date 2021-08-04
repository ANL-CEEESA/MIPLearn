#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import uniform, randint

from miplearn.problems.tsp import TravelingSalesmanGenerator, TravelingSalesmanInstance
from miplearn.solvers.learning import LearningSolver


def test_generator() -> None:
    instances = TravelingSalesmanGenerator(
        x=uniform(loc=0.0, scale=1000.0),
        y=uniform(loc=0.0, scale=1000.0),
        n=randint(low=100, high=101),
        gamma=uniform(loc=0.95, scale=0.1),
        fix_cities=True,
    ).generate(100)
    assert len(instances) == 100
    assert instances[0].n_cities == 100
    assert norm(instances[0].distances - instances[0].distances.T) < 1e-6
    d = [instance.distances[0, 1] for instance in instances]
    assert np.std(d) > 0


def test_instance() -> None:
    n_cities = 4
    distances = np.array(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
    )
    instance = TravelingSalesmanInstance(n_cities, distances)
    solver = LearningSolver()
    solver.solve(instance)
    assert len(instance.get_samples()) == 1
    sample = instance.get_samples()[0]
    assert sample.get_vector("mip_var_values") == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert sample.get_scalar("mip_lower_bound") == 4.0
    assert sample.get_scalar("mip_upper_bound") == 4.0


def test_subtour() -> None:
    n_cities = 6
    cities = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [3.0, 1.0],
        ]
    )
    distances = squareform(pdist(cities))
    instance = TravelingSalesmanInstance(n_cities, distances)
    solver = LearningSolver()
    solver.solve(instance)
    samples = instance.get_samples()
    assert len(samples) == 1
    sample = samples[0]
    lazy_enforced = sample.get_set("mip_constr_lazy_enforced")
    assert lazy_enforced is not None
    assert len(lazy_enforced) > 0
    assert sample.get_vector("mip_var_values") == [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    solver.fit([instance])
    solver.solve(instance)
