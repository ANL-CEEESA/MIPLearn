#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Any, List, Dict
from unittest.mock import Mock

from miplearn.components.cuts.mem import MemorizingCutsComponent
from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.problems.stab import build_stab_model_gurobipy, build_stab_model_pyomo
from miplearn.solvers.learning import LearningSolver
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Callable


def test_mem_component_gp(
    stab_gp_h5: List[str],
    stab_pyo_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    for h5 in [stab_pyo_h5, stab_gp_h5]:
        clf = Mock(wraps=DummyClassifier())
        comp = MemorizingCutsComponent(clf=clf, extractor=default_extractor)
        comp.fit(h5)

        # Should call fit method with correct arguments
        clf.fit.assert_called()
        x, y = clf.fit.call_args.args
        assert x.shape == (3, 50)
        assert y.shape == (3, 415)
        y = y.tolist()
        assert y[0][:20] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert y[1][:20] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert y[2][:20] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]

        # Should store violations
        assert comp.constrs_ is not None
        assert comp.n_features_ == 50
        assert comp.n_targets_ == 415
        assert len(comp.constrs_) == 415

        # Call before-mip
        stats: Dict[str, Any] = {}
        model = Mock()
        comp.before_mip(h5[0], model, stats)

        # Should call predict with correct args
        clf.predict.assert_called()
        (x_test,) = clf.predict.call_args.args
        assert x_test.shape == (1, 50)

        # Should set cuts_aot_
        assert model.cuts_aot_ is not None
        assert len(model.cuts_aot_) == 285


def test_usage_stab(
    stab_gp_h5: List[str],
    stab_pyo_h5: List[str],
    default_extractor: FeaturesExtractor,
) -> None:
    for (h5, build_model) in [
        (stab_pyo_h5, build_stab_model_pyomo),
        (stab_gp_h5, build_stab_model_gurobipy),
    ]:
        data_filenames = [f.replace(".h5", ".pkl.gz") for f in h5]
        clf = KNeighborsClassifier(n_neighbors=1)
        comp = MemorizingCutsComponent(clf=clf, extractor=default_extractor)
        solver = LearningSolver(components=[comp])
        solver.fit(data_filenames)
        stats = solver.optimize(data_filenames[0], build_model)  # type: ignore
        assert stats["Cuts: AOT"] > 0
