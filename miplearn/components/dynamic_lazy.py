#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import json
import logging
from typing import Dict, List, TYPE_CHECKING, Tuple, Any, Optional

import numpy as np
from overrides import overrides
from tqdm.auto import tqdm

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import MinProbabilityThreshold, Threshold
from miplearn.components.component import Component
from miplearn.components.dynamic_common import DynamicConstraintsComponent
from miplearn.features.sample import Sample, Hdf5Sample
from miplearn.instance.base import Instance
from miplearn.types import LearningSolveStats, ConstraintName, ConstraintCategory
from p_tqdm import p_map

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


class DynamicLazyConstraintsComponent(Component):
    """
    A component that predicts which lazy constraints to enforce.
    """

    def __init__(
        self,
        classifier: Classifier = CountingClassifier(),
        threshold: Threshold = MinProbabilityThreshold([0, 0.05]),
    ):
        self.dynamic: DynamicConstraintsComponent = DynamicConstraintsComponent(
            classifier=classifier,
            threshold=threshold,
            attr="mip_constr_lazy",
        )
        self.classifiers = self.dynamic.classifiers
        self.thresholds = self.dynamic.thresholds
        self.known_violations = self.dynamic.known_violations
        self.lazy_enforced: Dict[ConstraintName, Any] = {}
        self.n_iterations: int = 0

    @staticmethod
    def enforce(
        violations: Dict[ConstraintName, Any],
        instance: Instance,
        model: Any,
        solver: "LearningSolver",
    ) -> None:
        assert solver.internal_solver is not None
        for (vname, vdata) in violations.items():
            instance.enforce_lazy_constraint(solver.internal_solver, model, vdata)

    @overrides
    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        self.lazy_enforced.clear()
        logger.info("Predicting violated (dynamic) lazy constraints...")
        vnames = self.dynamic.sample_predict(instance, sample)
        violations = {c: self.dynamic.known_violations[c] for c in vnames}
        logger.info("Enforcing %d lazy constraints..." % len(vnames))
        self.enforce(violations, instance, model, solver)
        self.n_iterations = 0

    @overrides
    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        sample.put_scalar("mip_constr_lazy", self.dynamic.encode(self.lazy_enforced))
        stats["LazyDynamic: Added in callback"] = len(self.lazy_enforced)
        stats["LazyDynamic: Iterations"] = self.n_iterations

    @overrides
    def iteration_cb(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
    ) -> bool:
        assert solver.internal_solver is not None
        logger.debug("Finding violated lazy constraints...")
        violations = instance.find_violated_lazy_constraints(
            solver.internal_solver,
            model,
        )
        if len(violations) == 0:
            logger.debug("No violations found")
            return False
        else:
            self.n_iterations += 1
            for v in violations:
                self.lazy_enforced[v] = violations[v]
            logger.debug("    %d violations found" % len(violations))
            self.enforce(violations, instance, model, solver)
            return True

    # Delegate ML methods to self.dynamic
    # -------------------------------------------------------------------
    @overrides
    def sample_xy(
        self,
        instance: Optional[Instance],
        sample: Sample,
    ) -> Tuple[Dict, Dict]:
        return self.dynamic.sample_xy(instance, sample)

    @overrides
    def pre_fit(self, pre: List[Any]) -> None:
        self.dynamic.pre_fit(pre)

    def sample_predict(
        self,
        instance: Instance,
        sample: Sample,
    ) -> List[ConstraintName]:
        return self.dynamic.sample_predict(instance, sample)

    @overrides
    def pre_sample_xy(self, instance: Instance, sample: Sample) -> Any:
        return self.dynamic.pre_sample_xy(instance, sample)

    @overrides
    def fit_xy(
        self,
        x: Dict[ConstraintCategory, np.ndarray],
        y: Dict[ConstraintCategory, np.ndarray],
    ) -> None:
        self.dynamic.fit_xy(x, y)

    @overrides
    def sample_evaluate(
        self,
        instance: Instance,
        sample: Sample,
    ) -> Dict[ConstraintCategory, Dict[str, float]]:
        return self.dynamic.sample_evaluate(instance, sample)

    # ------------------------------------------------------------------------------------------------------------------
    # NEW API
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def extract(filenames, progress=True, known_cids=None):
        enforced_cids, features = [], []
        freeze_known_cids = True
        if known_cids is None:
            known_cids = set()
            freeze_known_cids = False
        for filename in tqdm(
            filenames,
            desc="extract (1/2)",
            disable=not progress,
        ):
            with Hdf5Sample(filename, mode="r") as sample:
                features.append(sample.get_array("lp_var_values"))
                cids = frozenset(
                    DynamicConstraintsComponent.decode(
                        sample.get_scalar("mip_constr_lazy")
                    ).keys()
                )
                enforced_cids.append(cids)
                if not freeze_known_cids:
                    known_cids.update(cids)

        x, y, cat, cdata = [], [], [], {}
        for (j, cid) in enumerate(known_cids):
            cdata[cid] = json.loads(cid.decode())
            for i in range(len(features)):
                cat.append(cid)
                x.append(features[i])
                if cid in enforced_cids[i]:
                    y.append([0, 1])
                else:
                    y.append([1, 0])
        x = np.vstack(x)
        y = np.vstack(y)
        cat = np.array(cat)
        x_dict, y_dict = DynamicLazyConstraintsComponent._split(
            x,
            y,
            cat,
            progress=progress,
        )
        return x_dict, y_dict, cdata

    @staticmethod
    def _split(x, y, cat, progress=False):
        # Sort data by categories
        pi = np.argsort(cat, kind="stable")
        x = x[pi]
        y = y[pi]
        cat = cat[pi]

        # Split categories
        x_dict = {}
        y_dict = {}
        start = 0
        for end in tqdm(
            range(len(cat) + 1),
            desc="extract (2/2)",
            disable=not progress,
        ):
            if (end >= len(cat)) or (cat[start] != cat[end]):
                x_dict[cat[start]] = x[start:end, :]
                y_dict[cat[start]] = y[start:end, :]
                start = end
        return x_dict, y_dict