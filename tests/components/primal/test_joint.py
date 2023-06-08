#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List, Dict, Any
from unittest.mock import Mock

from sklearn.dummy import DummyClassifier

from miplearn.components.primal.actions import SetWarmStart
from miplearn.components.primal.joint import JointVarsPrimalComponent
from miplearn.extractors.fields import H5FieldsExtractor


def test_joint(multiknapsack_h5: List[str]) -> None:
    # Create mock classifier
    clf = Mock(wraps=DummyClassifier())

    # Create and fit component
    comp = JointVarsPrimalComponent(
        clf=clf,
        extractor=H5FieldsExtractor(instance_fields=["static_var_obj_coeffs"]),
        action=SetWarmStart(),
    )
    comp.fit(multiknapsack_h5)

    # Should call fit method with correct arguments
    clf.fit.assert_called()
    x, y = clf.fit.call_args.args
    assert x.shape == (3, 100)
    assert y.shape == (3, 100)

    # Call before-mip
    stats: Dict[str, Any] = {}
    model = Mock()
    comp.before_mip(multiknapsack_h5[0], model, stats)

    # Should call predict with correct args
    clf.predict.assert_called()
    (x_test,) = clf.predict.call_args.args
    assert x_test.shape == (1, 100)

    # Should set warm starts
    model.set_warm_starts.assert_called()
    names, starts, _ = model.set_warm_starts.call_args.args
    assert len(names) == 100
    assert starts.shape == (1, 100)
