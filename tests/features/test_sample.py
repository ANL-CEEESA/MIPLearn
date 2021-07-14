#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn.features.sample import MemorySample, Sample


def _test_sample(sample: Sample) -> None:
    # Strings
    sample.put("str", "hello")
    assert sample.get("str") == "hello"

    # Numbers
    sample.put("int", 1)
    sample.put("float", 5.0)
    assert sample.get("int") == 1
    assert sample.get("float") == 5.0

    # List of strings
    sample.put("strlist", ["hello", "world"])
    assert sample.get("strlist") == ["hello", "world"]

    # List of numbers
    sample.put("intlist", [1, 2, 3])
    sample.put("floatlist", [4.0, 5.0, 6.0])
    assert sample.get("intlist") == [1, 2, 3]
    assert sample.get("floatlist") == [4.0, 5.0, 6.0]


def test_memory_sample() -> None:
    _test_sample(MemorySample())
