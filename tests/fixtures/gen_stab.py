from os.path import dirname

import numpy as np
from scipy.stats import uniform, randint

from miplearn.collectors.basic import BasicCollector
from miplearn.io import write_pkl_gz
from miplearn.problems.stab import (
    MaxWeightStableSetGenerator,
    build_stab_model_gurobipy,
    build_stab_model_pyomo,
)


np.random.seed(42)
gen = MaxWeightStableSetGenerator(
    w=uniform(10.0, scale=1.0),
    n=randint(low=50, high=51),
    p=uniform(loc=0.5, scale=0.0),
    fix_graph=True,
)
data = gen.generate(3)

params = {"seed": 42, "threads": 1}

# Gurobipy
data_filenames = write_pkl_gz(data, dirname(__file__), prefix="stab-gp-n50-")
collector = BasicCollector()
collector.collect(
    data_filenames,
    lambda data: build_stab_model_gurobipy(data, params=params),
    progress=True,
    verbose=True,
)

# Pyomo
data_filenames = write_pkl_gz(data, dirname(__file__), prefix="stab-pyo-n50-")
collector = BasicCollector()
collector.collect(
    data_filenames,
    lambda model: build_stab_model_pyomo(model, params=params),
    progress=True,
    verbose=True,
)
