#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import tempfile

from miplearn.solvers.learning import LearningSolver
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.features.sample import Hdf5Sample
from miplearn.instance.file import FileInstance


def test_usage() -> None:
    # Create original instance
    original = GurobiSolver().build_test_instance_knapsack()

    # Save instance to disk
    file = tempfile.NamedTemporaryFile()
    FileInstance.save(original, file.name)
    sample = Hdf5Sample(file.name)
    assert len(sample.get_bytes("pickled")) > 0

    # Solve instance from disk
    solver = LearningSolver(solver=GurobiSolver())
    solver.solve(FileInstance(file.name))

    # Assert HDF5 contains training data
    sample = FileInstance(file.name).get_samples()[0]
    assert sample.get_scalar("mip_lower_bound") == 1183.0
    assert sample.get_scalar("mip_upper_bound") == 1183.0
    assert len(sample.get_vector("lp_var_values")) == 5
    assert len(sample.get_vector("mip_var_values")) == 5
