#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import tempfile

from miplearn.instance import write_pickle_gz, PickleGzInstance
from miplearn.solvers.gurobi import GurobiSolver
from tests.fixtures.knapsack import get_knapsack_instance


def test_pickled() -> None:
    original = get_knapsack_instance(GurobiSolver())
    file = tempfile.NamedTemporaryFile()
    write_pickle_gz(original, file.name)
    pickled = PickleGzInstance(file.name)
    assert pickled.to_model() is not None
