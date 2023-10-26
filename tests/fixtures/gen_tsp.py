from os.path import dirname

import numpy as np
from scipy.stats import uniform, randint

from miplearn.collectors.basic import BasicCollector
from miplearn.io import write_pkl_gz
from miplearn.problems.tsp import TravelingSalesmanGenerator, build_tsp_model

np.random.seed(42)
gen = TravelingSalesmanGenerator(
    x=uniform(loc=0.0, scale=1000.0),
    y=uniform(loc=0.0, scale=1000.0),
    n=randint(low=20, high=21),
    gamma=uniform(loc=1.0, scale=0.25),
    fix_cities=True,
    round=True,
)
data = gen.generate(3)
data_filenames = write_pkl_gz(data, dirname(__file__), prefix="tsp-n20-")
collector = BasicCollector()
collector.collect(data_filenames, build_tsp_model)
