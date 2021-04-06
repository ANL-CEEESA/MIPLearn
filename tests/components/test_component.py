#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from unittest.mock import Mock

from miplearn.components.component import Component
from miplearn.instance.base import Instance


def test_xy_instance():
    def _sample_xy(features, sample):
        x = {
            "s1": {
                "category_a": [
                    [1, 2, 3],
                    [3, 4, 6],
                ],
                "category_b": [
                    [7, 8, 9],
                ],
            },
            "s2": {
                "category_a": [
                    [0, 0, 0],
                    [0, 5, 3],
                    [2, 2, 0],
                ],
                "category_c": [
                    [0, 0, 0],
                    [0, 0, 1],
                ],
            },
            "s3": {
                "category_c": [
                    [1, 1, 1],
                ],
            },
        }
        y = {
            "s1": {
                "category_a": [[1], [2]],
                "category_b": [[3]],
            },
            "s2": {
                "category_a": [[4], [5], [6]],
                "category_c": [[8], [9], [10]],
            },
            "s3": {
                "category_c": [[11]],
            },
        }
        return x[sample], y[sample]

    comp = Component()
    instance_1 = Mock(spec=Instance)
    instance_1.training_data = ["s1", "s2"]
    instance_1.features = {}
    instance_2 = Mock(spec=Instance)
    instance_2.training_data = ["s3"]
    instance_2.features = {}
    comp.sample_xy = _sample_xy
    x_expected = {
        "category_a": [
            [1, 2, 3],
            [3, 4, 6],
            [0, 0, 0],
            [0, 5, 3],
            [2, 2, 0],
        ],
        "category_b": [
            [7, 8, 9],
        ],
        "category_c": [
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
    }
    y_expected = {
        "category_a": [
            [1],
            [2],
            [4],
            [5],
            [6],
        ],
        "category_b": [
            [3],
        ],
        "category_c": [
            [8],
            [9],
            [10],
            [11],
        ],
    }
    x_actual, y_actual = comp.xy_instances([instance_1, instance_2])
    assert x_actual == x_expected
    assert y_actual == y_expected
