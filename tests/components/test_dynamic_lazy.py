#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List, cast
from unittest.mock import Mock

import numpy as np
import pytest

from miplearn.classifiers import Classifier
from miplearn.classifiers.threshold import MinProbabilityThreshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.dynamic_lazy import DynamicLazyConstraintsComponent
from miplearn.features.sample import Sample, MemorySample
from miplearn.instance.base import Instance
from miplearn.solvers.tests import assert_equals

E = 0.1


@pytest.fixture
def training_instances() -> List[Instance]:
    instances = [cast(Instance, Mock(spec=Instance)) for _ in range(2)]
    samples_0 = [
        MemorySample(
            {
                "lazy_enforced": {"c1", "c2"},
                "instance_features_user": [5.0],
            },
        ),
        MemorySample(
            {
                "lazy_enforced": {"c2", "c3"},
                "instance_features_user": [5.0],
            },
        ),
    ]
    instances[0].get_samples = Mock(return_value=samples_0)  # type: ignore
    instances[0].get_constraint_categories = Mock(  # type: ignore
        return_value={
            "c1": "type-a",
            "c2": "type-a",
            "c3": "type-b",
            "c4": "type-b",
        }
    )
    instances[0].get_constraint_features = Mock(  # type: ignore
        return_value={
            "c1": [1.0, 2.0, 3.0],
            "c2": [4.0, 5.0, 6.0],
            "c3": [1.0, 2.0],
            "c4": [3.0, 4.0],
        }
    )
    samples_1 = [
        MemorySample(
            {
                "lazy_enforced": {"c3", "c4"},
                "instance_features_user": [8.0],
            },
        )
    ]
    instances[1].get_samples = Mock(return_value=samples_1)  # type: ignore
    instances[1].get_constraint_categories = Mock(  # type: ignore
        return_value={
            "c1": None,
            "c2": "type-a",
            "c3": "type-b",
            "c4": "type-b",
        }
    )
    instances[1].get_constraint_features = Mock(  # type: ignore
        return_value={
            "c2": [7.0, 8.0, 9.0],
            "c3": [5.0, 6.0],
            "c4": [7.0, 8.0],
        }
    )
    return instances


def test_sample_xy(training_instances: List[Instance]) -> None:
    comp = DynamicLazyConstraintsComponent()
    comp.pre_fit([{"c1", "c2", "c3", "c4"}])
    x_expected = {
        "type-a": [[5.0, 1.0, 2.0, 3.0], [5.0, 4.0, 5.0, 6.0]],
        "type-b": [[5.0, 1.0, 2.0], [5.0, 3.0, 4.0]],
    }
    y_expected = {
        "type-a": [[False, True], [False, True]],
        "type-b": [[True, False], [True, False]],
    }
    x_actual, y_actual = comp.sample_xy(
        training_instances[0],
        training_instances[0].get_samples()[0],
    )
    assert_equals(x_actual, x_expected)
    assert_equals(y_actual, y_expected)


# def test_fit(training_instances: List[Instance]) -> None:
#     clf = Mock(spec=Classifier)
#     clf.clone = Mock(side_effect=lambda: Mock(spec=Classifier))
#     comp = DynamicLazyConstraintsComponent(classifier=clf)
#     comp.fit(training_instances)
#     assert clf.clone.call_count == 2
#
#     assert "type-a" in comp.classifiers
#     clf_a = comp.classifiers["type-a"]
#     assert clf_a.fit.call_count == 1  # type: ignore
#     assert_array_equal(
#         clf_a.fit.call_args[0][0],  # type: ignore
#         np.array(
#             [
#                 [5.0, 1.0, 2.0, 3.0],
#                 [5.0, 4.0, 5.0, 6.0],
#                 [5.0, 1.0, 2.0, 3.0],
#                 [5.0, 4.0, 5.0, 6.0],
#                 [8.0, 7.0, 8.0, 9.0],
#             ]
#         ),
#     )
#     assert_array_equal(
#         clf_a.fit.call_args[0][1],  # type: ignore
#         np.array(
#             [
#                 [False, True],
#                 [False, True],
#                 [True, False],
#                 [False, True],
#                 [True, False],
#             ]
#         ),
#     )
#
#     assert "type-b" in comp.classifiers
#     clf_b = comp.classifiers["type-b"]
#     assert clf_b.fit.call_count == 1  # type: ignore
#     assert_array_equal(
#         clf_b.fit.call_args[0][0],  # type: ignore
#         np.array(
#             [
#                 [5.0, 1.0, 2.0],
#                 [5.0, 3.0, 4.0],
#                 [5.0, 1.0, 2.0],
#                 [5.0, 3.0, 4.0],
#                 [8.0, 5.0, 6.0],
#                 [8.0, 7.0, 8.0],
#             ]
#         ),
#     )
#     assert_array_equal(
#         clf_b.fit.call_args[0][1],  # type: ignore
#         np.array(
#             [
#                 [True, False],
#                 [True, False],
#                 [False, True],
#                 [True, False],
#                 [False, True],
#                 [False, True],
#             ]
#         ),
#     )


def test_sample_predict_evaluate(training_instances: List[Instance]) -> None:
    comp = DynamicLazyConstraintsComponent()
    comp.known_cids.extend(["c1", "c2", "c3", "c4"])
    comp.thresholds["type-a"] = MinProbabilityThreshold([0.5, 0.5])
    comp.thresholds["type-b"] = MinProbabilityThreshold([0.5, 0.5])
    comp.classifiers["type-a"] = Mock(spec=Classifier)
    comp.classifiers["type-b"] = Mock(spec=Classifier)
    comp.classifiers["type-a"].predict_proba = Mock(  # type: ignore
        side_effect=lambda _: np.array([[0.1, 0.9], [0.8, 0.2]])
    )
    comp.classifiers["type-b"].predict_proba = Mock(  # type: ignore
        side_effect=lambda _: np.array([[0.9, 0.1], [0.1, 0.9]])
    )
    pred = comp.sample_predict(
        training_instances[0],
        training_instances[0].get_samples()[0],
    )
    assert pred == ["c1", "c4"]
    ev = comp.sample_evaluate(
        training_instances[0],
        training_instances[0].get_samples()[0],
    )
    assert ev == {
        "type-a": classifier_evaluation_dict(tp=1, fp=0, tn=0, fn=1),
        "type-b": classifier_evaluation_dict(tp=0, fp=1, tn=1, fn=0),
    }
