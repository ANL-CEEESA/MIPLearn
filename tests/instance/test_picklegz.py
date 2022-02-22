#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import tempfile
from typing import cast, IO

from miplearn.instance.picklegz import write_pickle_gz, PickleGzInstance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn import save
from os.path import exists
import gzip
import pickle


def test_usage() -> None:
    original = GurobiSolver().build_test_instance_knapsack()
    file = tempfile.NamedTemporaryFile()
    write_pickle_gz(original, file.name)
    pickled = PickleGzInstance(file.name)
    pickled.load()
    assert pickled.to_model() is not None


def test_save() -> None:
    objs = [1, "ABC", True]
    with tempfile.TemporaryDirectory() as dirname:
        filenames = save(objs, dirname)
        assert len(filenames) == 3
        for (idx, f) in enumerate(filenames):
            assert exists(f)
            with gzip.GzipFile(f, "rb") as file:
                assert pickle.load(cast(IO[bytes], file)) == objs[idx]
