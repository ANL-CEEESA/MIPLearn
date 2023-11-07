from os.path import dirname

import numpy as np
from scipy.stats import uniform, randint

from miplearn.collectors.basic import BasicCollector
from miplearn.io import write_pkl_gz
from miplearn.problems.stab import (
    MaxWeightStableSetGenerator,
    build_stab_model,
)

np.random.seed(42)
gen = MaxWeightStableSetGenerator(
    w=uniform(10.0, scale=1.0),
    n=randint(low=50, high=51),
    p=uniform(loc=0.5, scale=0.0),
    fix_graph=True,
)
data = gen.generate(3)
data_filenames = write_pkl_gz(data, dirname(__file__), prefix="stab-n50-")
collector = BasicCollector()
collector.collect(data_filenames, build_stab_model)
