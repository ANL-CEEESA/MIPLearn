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
from miplearn.components.dynamic_common import DynamicConstraintsComponent
from miplearn.components.dynamic_lazy import DynamicLazyConstraintsComponent
from miplearn.features.sample import MemorySample
from miplearn.instance.base import Instance
from miplearn.solvers.tests import assert_equals

E = 0.1


@pytest.fixture
def training_instances() -> List[Instance]:
    instances = [cast(Instance, Mock(spec=Instance)) for _ in range(2)]
    samples_0 = [
        MemorySample(
            {
                "mip_constr_lazy": DynamicConstraintsComponent.encode(
                    {
                        b"c1": 0,
                        b"c2": 0,
                    }
                ),
                "static_instance_features": np.array([5.0]),
            },
        ),
        MemorySample(
            {
                "mip_constr_lazy": DynamicConstraintsComponent.encode(
                    {
                        b"c2": 0,
                        b"c3": 0,
                    }
                ),
                "static_instance_features": np.array([5.0]),
            },
        ),
    ]
    instances[0].get_samples = Mock(return_value=samples_0)  # type: ignore
    instances[0].get_constraint_categories = Mock(  # type: ignore
        return_value=np.array(["type-a", "type-a", "type-b", "type-b"], dtype="S")
    )
    instances[0].get_constraint_features = Mock(  # type: ignore
        return_value=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
            ]
        )
    )
    instances[0].are_constraints_lazy = Mock(  # type: ignore
        return_value=np.zeros(4, dtype=bool)
    )
    samples_1 = [
        MemorySample(
            {
                "mip_constr_lazy": DynamicConstraintsComponent.encode(
                    {
                        b"c3": 0,
                        b"c4": 0,
                    }
                ),
                "static_instance_features": np.array([8.0]),
            },
        )
    ]
    instances[1].get_samples = Mock(return_value=samples_1)  # type: ignore
    instances[1].get_constraint_categories = Mock(  # type: ignore
        return_value=np.array(["", "type-a", "type-b", "type-b"], dtype="S")
    )
    instances[1].get_constraint_features = Mock(  # type: ignore
        return_value=np.array(
            [
                [7.0, 8.0, 9.0],
                [5.0, 6.0, 0.0],
                [7.0, 8.0, 0.0],
            ]
        )
    )
    instances[1].are_constraints_lazy = Mock(  # type: ignore
        return_value=np.zeros(4, dtype=bool)
    )
    return instances


def test_sample_xy(training_instances: List[Instance]) -> None:
    comp = DynamicLazyConstraintsComponent()
    comp.pre_fit(
        [
            {b"c1": 0, b"c3": 0, b"c4": 0},
            {b"c1": 0, b"c2": 0, b"c4": 0},
        ]
    )
    x_expected = {
        b"type-a": np.array([[5.0, 1.0, 2.0, 3.0], [5.0, 4.0, 5.0, 6.0]]),
        b"type-b": np.array([[5.0, 1.0, 2.0, 0.0], [5.0, 3.0, 4.0, 0.0]]),
    }
    y_expected = {
        b"type-a": np.array([[False, True], [False, True]]),
        b"type-b": np.array([[True, False], [True, False]]),
    }
    x_actual, y_actual = comp.sample_xy(
        training_instances[0],
        training_instances[0].get_samples()[0],
    )
    assert_equals(x_actual, x_expected)
    assert_equals(y_actual, y_expected)


def test_sample_predict_evaluate(training_instances: List[Instance]) -> None:
    comp = DynamicLazyConstraintsComponent()
    comp.known_violations[b"c1"] = 0
    comp.known_violations[b"c2"] = 0
    comp.known_violations[b"c3"] = 0
    comp.known_violations[b"c4"] = 0
    comp.thresholds[b"type-a"] = MinProbabilityThreshold([0.5, 0.5])
    comp.thresholds[b"type-b"] = MinProbabilityThreshold([0.5, 0.5])
    comp.classifiers[b"type-a"] = Mock(spec=Classifier)
    comp.classifiers[b"type-b"] = Mock(spec=Classifier)
    comp.classifiers[b"type-a"].predict_proba = Mock(  # type: ignore
        side_effect=lambda _: np.array([[0.1, 0.9], [0.8, 0.2]])
    )
    comp.classifiers[b"type-b"].predict_proba = Mock(  # type: ignore
        side_effect=lambda _: np.array([[0.9, 0.1], [0.1, 0.9]])
    )
    pred = comp.sample_predict(
        training_instances[0],
        training_instances[0].get_samples()[0],
    )
    assert pred == [b"c1", b"c4"]
    ev = comp.sample_evaluate(
        training_instances[0],
        training_instances[0].get_samples()[0],
    )
    assert ev == classifier_evaluation_dict(tp=1, fp=1, tn=1, fn=1)
