#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import List, Dict, Any
from unittest.mock import Mock

from miplearn.components.primal.actions import SetWarmStart, FixVariables
from miplearn.components.primal.expert import ExpertPrimalComponent


def test_expert(multiknapsack_h5: List[str]) -> None:
    model = Mock()
    stats: Dict[str, Any] = {}
    comp = ExpertPrimalComponent(action=SetWarmStart())
    comp.before_mip(multiknapsack_h5[0], model, stats)
    model.set_warm_starts.assert_called()
    names, starts, _ = model.set_warm_starts.call_args.args
    assert names.shape == (100,)
    assert starts.shape == (1, 100)

    comp = ExpertPrimalComponent(action=FixVariables())
    comp.before_mip(multiknapsack_h5[0], model, stats)
    model.fix_variables.assert_called()
    names, v, _ = model.fix_variables.call_args.args
    assert names.shape == (100,)
    assert v.shape == (100,)
