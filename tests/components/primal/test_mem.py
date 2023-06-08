#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from typing import List, Dict, Any
from unittest.mock import Mock

import numpy as np
from sklearn.dummy import DummyClassifier

from miplearn.components.primal.actions import SetWarmStart
from miplearn.components.primal.mem import (
    MemorizingPrimalComponent,
    SelectTopSolutions,
    MergeTopSolutions,
)
from miplearn.extractors.abstract import FeaturesExtractor

logger = logging.getLogger(__name__)


def test_mem_component(
    multiknapsack_h5: List[str], default_extractor: FeaturesExtractor
) -> None:
    # Create mock classifier
    clf = Mock(wraps=DummyClassifier())

    # Create and fit component
    comp = MemorizingPrimalComponent(
        clf,
        extractor=default_extractor,
        constructor=SelectTopSolutions(2),
        action=SetWarmStart(),
    )
    comp.fit(multiknapsack_h5)

    # Should call fit method with correct arguments
    clf.fit.assert_called()
    x, y = clf.fit.call_args.args
    assert x.shape == (3, 100)
    assert y.tolist() == [0, 1, 2]

    # Should store solutions
    assert comp.solutions_ is not None
    assert comp.solutions_.shape == (3, 100)
    assert comp.bin_var_names_ is not None
    assert len(comp.bin_var_names_) == 100

    # Call before-mip
    stats: Dict[str, Any] = {}
    model = Mock()
    comp.before_mip(multiknapsack_h5[0], model, stats)

    # Should call predict_proba with correct args
    clf.predict_proba.assert_called()
    (x_test,) = clf.predict_proba.call_args.args
    assert x_test.shape == (1, 100)

    # Should set warm starts
    model.set_warm_starts.assert_called()
    names, starts, _ = model.set_warm_starts.call_args.args
    assert len(names) == 100
    assert starts.shape == (2, 100)
    assert np.all(starts[0, :] == comp.solutions_[0, :])
    assert np.all(starts[1, :] == comp.solutions_[1, :])


def test_merge_top_solutions() -> None:
    solutions = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    y_proba = np.array([0.25, 0.25, 0.25, 0.25, 0])
    starts = MergeTopSolutions(k=4, thresholds=[0.25, 0.75]).construct(
        y_proba, solutions
    )
    assert starts.shape == (1, 4)
    assert starts[0, 0] == 0
    assert starts[0, 1] == 1
    assert np.isnan(starts[0, 2])
    assert starts[0, 3] == 1
