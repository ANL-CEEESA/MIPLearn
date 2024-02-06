#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import List, Dict, Any
from unittest.mock import Mock

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from miplearn.components.lazy.mem import MemorizingLazyComponent
from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.problems.tsp import build_tsp_model_gurobipy, build_tsp_model_pyomo
from miplearn.solvers.learning import LearningSolver


def test_mem_component(
    tsp_gp_h5: List[str],
    tsp_pyo_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    for h5 in [tsp_gp_h5, tsp_pyo_h5]:
        clf = Mock(wraps=DummyClassifier())
        comp = MemorizingLazyComponent(clf=clf, extractor=default_extractor)
        comp.fit(tsp_gp_h5)

        # Should call fit method with correct arguments
        clf.fit.assert_called()
        x, y = clf.fit.call_args.args
        assert x.shape == (3, 190)
        assert y.tolist() == [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        ]

        # Should store violations
        assert comp.constrs_ is not None
        assert comp.n_features_ == 190
        assert comp.n_targets_ == 20
        assert len(comp.constrs_) == 20

        # Call before-mip
        stats: Dict[str, Any] = {}
        model = Mock()
        comp.before_mip(tsp_gp_h5[0], model, stats)

        # Should call predict with correct args
        clf.predict.assert_called()
        (x_test,) = clf.predict.call_args.args
        assert x_test.shape == (1, 190)


def test_usage_tsp(
    tsp_gp_h5: List[str],
    tsp_pyo_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    for h5, build_model in [
        (tsp_pyo_h5, build_tsp_model_pyomo),
        (tsp_gp_h5, build_tsp_model_gurobipy),
    ]:
        data_filenames = [f.replace(".h5", ".pkl.gz") for f in h5]
        clf = KNeighborsClassifier(n_neighbors=1)
        comp = MemorizingLazyComponent(clf=clf, extractor=default_extractor)
        solver = LearningSolver(components=[comp])
        solver.fit(data_filenames)
        stats = solver.optimize(data_filenames[0], build_model)  # type: ignore
        assert stats["Lazy Constraints: AOT"] > 0
