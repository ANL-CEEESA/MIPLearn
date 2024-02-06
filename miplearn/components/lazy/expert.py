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


class ExpertLazyComponent:
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
            violations_str = h5.get_scalar("mip_lazy")
            assert violations_str is not None
            assert isinstance(violations_str, str)
            violations = list(set(convert_lists_to_tuples(json.loads(violations_str))))
            logger.info(f"Enforcing {len(violations)} constraints ahead-of-time...")
            model.lazy_enforce(violations)
            stats["Lazy Constraints: AOT"] = len(violations)
