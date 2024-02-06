#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import json
import logging
from typing import Dict, Any, List

from miplearn.components.cuts.mem import convert_lists_to_tuples
from miplearn.h5 import H5File
from miplearn.solvers.abstract import AbstractModel

logger = logging.getLogger(__name__)


class ExpertCutsComponent:
    def fit(
        self,
        _: List[str],
    ) -> None:
        pass

    def before_mip(
        self,
        test_h5: str,
        model: AbstractModel,
        stats: Dict[str, Any],
    ) -> None:
        with H5File(test_h5, "r") as h5:
            cuts_str = h5.get_scalar("mip_cuts")
            assert cuts_str is not None
            assert isinstance(cuts_str, str)
            cuts = list(set(convert_lists_to_tuples(json.loads(cuts_str))))
            model.set_cuts(cuts)
            stats["Cuts: AOT"] = len(cuts)
