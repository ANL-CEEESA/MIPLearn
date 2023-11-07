#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any, List, Hashable, Dict
from unittest.mock import Mock

import gurobipy as gp
import networkx as nx
from gurobipy import GRB, quicksum
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from miplearn.components.cuts.mem import MemorizingCutsComponent
from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.problems.stab import build_stab_model
from miplearn.solvers.gurobi import GurobiModel
from miplearn.solvers.learning import LearningSolver
import numpy as np


# def test_usage() -> None:
#     model = _build_cut_model()
#     solver = LearningSolver(components=[])
#     solver.optimize(model)
#     assert model.cuts_ is not None
#     assert len(model.cuts_) > 0
#     assert False


def test_mem_component(
    stab_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    clf = Mock(wraps=DummyClassifier())
    comp = MemorizingCutsComponent(clf=clf, extractor=default_extractor)
    comp.fit(stab_h5)

    # Should call fit method with correct arguments
    clf.fit.assert_called()
    x, y = clf.fit.call_args.args
    assert x.shape == (3, 50)
    assert y.shape == (3, 388)
    y = y.tolist()
    assert y[0][:20] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert y[1][:20] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    assert y[2][:20] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    # Should store violations
    assert comp.constrs_ is not None
    assert comp.n_features_ == 50
    assert comp.n_targets_ == 388
    assert len(comp.constrs_) == 388

    # Call before-mip
    stats: Dict[str, Any] = {}
    model = Mock()
    comp.before_mip(stab_h5[0], model, stats)

    # Should call predict with correct args
    clf.predict.assert_called()
    (x_test,) = clf.predict.call_args.args
    assert x_test.shape == (1, 50)

    # Should set cuts_aot_
    assert model.cuts_aot_ is not None
    assert len(model.cuts_aot_) == 243


def test_usage_stab(
    stab_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    data_filenames = [f.replace(".h5", ".pkl.gz") for f in stab_h5]
    clf = KNeighborsClassifier(n_neighbors=1)
    comp = MemorizingCutsComponent(clf=clf, extractor=default_extractor)
    solver = LearningSolver(components=[comp])
    solver.fit(data_filenames)
    stats = solver.optimize(data_filenames[0], build_stab_model)
    assert stats["Cuts: AOT"] > 0
