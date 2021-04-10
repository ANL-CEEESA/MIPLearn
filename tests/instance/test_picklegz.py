#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import tempfile

from miplearn.instance.picklegz import write_pickle_gz, PickleGzInstance
from miplearn.solvers.gurobi import GurobiSolver


def test_usage() -> None:
    original = GurobiSolver().build_test_instance_knapsack()
    file = tempfile.NamedTemporaryFile()
    write_pickle_gz(original, file.name)
    pickled = PickleGzInstance(file.name)
    pickled.load()
    assert pickled.to_model() is not None
