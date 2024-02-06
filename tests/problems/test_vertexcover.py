#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import networkx as nx
import numpy as np

from miplearn.problems.vertexcover import (
    MinWeightVertexCoverData,
    build_vertexcover_model_gurobipy,
)


def test_stab() -> None:
    data = MinWeightVertexCoverData(
        graph=nx.cycle_graph(5),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    model = build_vertexcover_model_gurobipy(data)
    model.optimize()
    assert model.inner.objVal == 3.0
