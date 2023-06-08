#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List, Dict, Any
from unittest.mock import Mock, call

from sklearn.dummy import DummyClassifier

from miplearn.components.primal.actions import SetWarmStart
from miplearn.components.primal.indep import IndependentVarsPrimalComponent
from miplearn.extractors.fields import H5FieldsExtractor


def test_indep(multiknapsack_h5: List[str]) -> None:
    # Create and fit component
    clone_fn = Mock(return_value=Mock(wraps=DummyClassifier()))
    comp = IndependentVarsPrimalComponent(
        base_clf="dummy",
        extractor=H5FieldsExtractor(var_fields=["lp_var_values"]),
        clone_fn=clone_fn,
        action=SetWarmStart(),
    )
    comp.fit(multiknapsack_h5)

    # Should call clone 100 times and store the 100 classifiers
    clone_fn.assert_has_calls([call("dummy") for _ in range(100)])
    assert len(comp.clf_) == 100

    for v in [b"x[0]", b"x[1]"]:
        # Should pass correct data to fit
        comp.clf_[v].fit.assert_called()
        x, y = comp.clf_[v].fit.call_args.args
        assert x.shape == (3, 1)
        assert y.shape == (3,)

    # Call before-mip
    stats: Dict[str, Any] = {}
    model = Mock()
    comp.before_mip(multiknapsack_h5[0], model, stats)

    # Should call predict with correct args
    for v in [b"x[0]", b"x[1]"]:
        comp.clf_[v].predict.assert_called()
        (x_test,) = comp.clf_[v].predict.call_args.args
        assert x_test.shape == (1, 1)

    # Should set warm starts
    model.set_warm_starts.assert_called()
    names, starts, _ = model.set_warm_starts.call_args.args
    assert len(names) == 100
    assert starts.shape == (1, 100)
