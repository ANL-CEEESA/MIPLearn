#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import LearningSolver
from miplearn.problems.tsp import TravelingSalesmanGenerator, TravelingSalesmanInstance
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import uniform, randint


def test_generator():
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


def test_instance():
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
    for solver_name in ["gurobi"]:
        solver = LearningSolver(solver=solver_name)
        solver.solve(instance)
        x = instance.solution["x"]
        assert x[0, 1] == 1.0
        assert x[0, 2] == 0.0
        assert x[0, 3] == 1.0
        assert x[1, 2] == 1.0
        assert x[1, 3] == 0.0
        assert x[2, 3] == 1.0
        assert instance.lower_bound == 4.0
        assert instance.upper_bound == 4.0


def test_subtour():
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
    for solver_name in ["gurobi"]:
        solver = LearningSolver(solver=solver_name)
        solver.solve(instance)
        assert hasattr(instance, "found_violated_lazy_constraints")
        assert hasattr(instance, "found_violated_user_cuts")
        x = instance.solution["x"]
        assert x[0, 1] == 1.0
        assert x[0, 4] == 1.0
        assert x[1, 2] == 1.0
        assert x[2, 3] == 1.0
        assert x[3, 5] == 1.0
        assert x[4, 5] == 1.0
        solver.fit([instance])
        solver.solve(instance)
