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
    stats = solver.solve(instance)
    solution = instance.training_data[0].solution
    assert solution is not None
    assert solution["x[(0, 1)]"] == 1.0
    assert solution["x[(0, 2)]"] == 0.0
    assert solution["x[(0, 3)]"] == 1.0
    assert solution["x[(1, 2)]"] == 1.0
    assert solution["x[(1, 3)]"] == 0.0
    assert solution["x[(2, 3)]"] == 1.0
    assert stats["Lower bound"] == 4.0
    assert stats["Upper bound"] == 4.0


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
    assert instance.training_data[0].lazy_enforced is not None
    assert len(instance.training_data[0].lazy_enforced) > 0
    solution = instance.training_data[0].solution
    assert solution is not None
    assert solution["x[(0, 1)]"] == 1.0
    assert solution["x[(0, 4)]"] == 1.0
    assert solution["x[(1, 2)]"] == 1.0
    assert solution["x[(2, 3)]"] == 1.0
    assert solution["x[(3, 5)]"] == 1.0
    assert solution["x[(4, 5)]"] == 1.0
    solver.fit([instance])
    solver.solve(instance)
