#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import List, Dict, Any, Hashable

from miplearn.components.cuts.mem import (
    _BaseMemorizingConstrComponent,
)
from miplearn.extractors.abstract import FeaturesExtractor
from miplearn.solvers.abstract import AbstractModel

logger = logging.getLogger(__name__)


class MemorizingLazyComponent(_BaseMemorizingConstrComponent):
    def __init__(self, clf: Any, extractor: FeaturesExtractor) -> None:
        super().__init__(clf, extractor, "mip_lazy")

    def before_mip(
        self,
        test_h5: str,
        model: AbstractModel,
        stats: Dict[str, Any],
    ) -> None:
        if model.lazy_enforce is None:
            return
        assert self.constrs_ is not None
        violations = self.predict("Predicting violated lazy constraints...", test_h5)
        logger.info(f"Enforcing {len(violations)} constraints ahead-of-time...")
        model.lazy_enforce(model, violations)
        stats["Lazy Constraints: AOT"] = len(violations)
