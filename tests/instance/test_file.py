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
    filename = tempfile.mktemp()
    FileInstance.save(original, filename)
    sample = Hdf5Sample(filename, check_data=True)
    assert len(sample.get_bytes("pickled")) > 0

    # Solve instance from disk
    solver = LearningSolver(solver=GurobiSolver())
    solver.solve(FileInstance(filename))

    # Assert HDF5 contains training data
    sample = FileInstance(filename).get_samples()[0]
    assert sample.get_scalar("mip_lower_bound") == 1183.0
    assert sample.get_scalar("mip_upper_bound") == 1183.0
    assert len(sample.get_array("lp_var_values")) == 5
    assert len(sample.get_array("mip_var_values")) == 5
