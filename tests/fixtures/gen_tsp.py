from os.path import dirname

import numpy as np
from scipy.stats import uniform, randint

from miplearn.collectors.basic import BasicCollector
from miplearn.io import write_pkl_gz
from miplearn.problems.tsp import (
    TravelingSalesmanGenerator,
    TravelingSalesmanPerturber,
    build_tsp_model_gurobipy,
    build_tsp_model_pyomo,
)

np.random.seed(42)
gen = TravelingSalesmanGenerator(
    x=uniform(loc=0.0, scale=1000.0),
    y=uniform(loc=0.0, scale=1000.0),
    n=randint(low=20, high=21),
    gamma=uniform(loc=1.0, scale=0.0),
    round=True,
)

# Generate a reference instance with fixed cities
reference_instance = gen.generate(1)[0]

# Generate perturbed instances with same cities but different distance scaling
perturber = TravelingSalesmanPerturber(
    gamma=uniform(loc=1.0, scale=0.25),
    round=True,
)
data = perturber.perturb(reference_instance, 3)

params = {"seed": 42, "threads": 1}

# Gurobipy
data_filenames = write_pkl_gz(data, dirname(__file__), prefix="tsp-gp-n20-")
collector = BasicCollector()
collector.collect(
    data_filenames,
    lambda d: build_tsp_model_gurobipy(d, params=params),
    progress=True,
    verbose=True,
)

# Pyomo
data_filenames = write_pkl_gz(data, dirname(__file__), prefix="tsp-pyo-n20-")
collector = BasicCollector()
collector.collect(
    data_filenames,
    lambda d: build_tsp_model_pyomo(d, params=params),
    progress=True,
    verbose=True,
)
