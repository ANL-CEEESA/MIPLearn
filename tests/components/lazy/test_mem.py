#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import List, Dict, Any
from unittest.mock import Mock

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from miplearn.components.lazy.mem import MemorizingLazyConstrComponent
from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.problems.tsp import build_tsp_model
from miplearn.solvers.learning import LearningSolver


def test_mem_component(
    tsp_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    clf = Mock(wraps=DummyClassifier())
    comp = MemorizingLazyConstrComponent(clf=clf, extractor=default_extractor)
    comp.fit(tsp_h5)

    # Should call fit method with correct arguments
    clf.fit.assert_called()
    x, y = clf.fit.call_args.args
    assert x.shape == (3, 190)
    assert y.tolist() == [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    ]

    # Should store violations
    assert comp.constrs_ is not None
    assert comp.n_features_ == 190
    assert comp.n_targets_ == 22
    assert len(comp.constrs_) == 22

    # Call before-mip
    stats: Dict[str, Any] = {}
    model = Mock()
    comp.before_mip(tsp_h5[0], model, stats)

    # Should call predict with correct args
    clf.predict.assert_called()
    (x_test,) = clf.predict.call_args.args
    assert x_test.shape == (1, 190)


def test_usage_tsp(
    tsp_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    # Should not crash
    data_filenames = [f.replace(".h5", ".pkl.gz") for f in tsp_h5]
    clf = KNeighborsClassifier(n_neighbors=1)
    comp = MemorizingLazyConstrComponent(clf=clf, extractor=default_extractor)
    solver = LearningSolver(components=[comp])
    solver.fit(data_filenames)
    solver.optimize(data_filenames[0], build_tsp_model)
