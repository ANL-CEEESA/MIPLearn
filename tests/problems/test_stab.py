#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from tempfile import NamedTemporaryFile

import networkx as nx
import numpy as np

from miplearn.h5 import H5File
from miplearn.problems.stab import (
    MaxWeightStableSetData,
    build_stab_model_pyomo,
    build_stab_model_gurobipy,
)


def test_stab() -> None:
    data = MaxWeightStableSetData(
        graph=nx.cycle_graph(5),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    for model in [
        build_stab_model_pyomo(data),
        build_stab_model_gurobipy(data),
    ]:
        with NamedTemporaryFile() as tempfile:
            with H5File(tempfile.name) as h5:
                model.optimize()
                model.extract_after_mip(h5)
                assert h5.get_scalar("mip_obj_value") == -2.0
