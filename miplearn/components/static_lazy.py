#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Dict, Tuple, List, Hashable, Any, TYPE_CHECKING, Set, Optional

import numpy as np
from overrides import overrides

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import MinProbabilityThreshold, Threshold
from miplearn.components.component import Component
from miplearn.features import Sample, ConstraintFeatures
from miplearn.instance.base import Instance
from miplearn.types import LearningSolveStats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


class LazyConstraint:
    def __init__(self, cid: str, obj: Any) -> None:
        self.cid = cid
        self.obj = obj


class StaticLazyConstraintsComponent(Component):
    """
    Component that decides which of the constraints tagged as lazy should
    be kept in the formulation, and which should be removed.
    """

    def __init__(
        self,
        classifier: Classifier = CountingClassifier(),
        threshold: Threshold = MinProbabilityThreshold([0.50, 0.50]),
        violation_tolerance: float = -0.5,
    ) -> None:
        assert isinstance(classifier, Classifier)
        self.classifier_prototype: Classifier = classifier
        self.threshold_prototype: Threshold = threshold
        self.classifiers: Dict[Hashable, Classifier] = {}
        self.thresholds: Dict[Hashable, Threshold] = {}
        self.pool: ConstraintFeatures = ConstraintFeatures()
        self.violation_tolerance: float = violation_tolerance
        self.enforced_cids: Set[Hashable] = set()
        self.n_restored: int = 0
        self.n_iterations: int = 0

    @overrides
    def after_solve_mip(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        assert sample.after_mip is not None
        assert sample.after_mip.extra is not None
        sample.after_mip.extra["lazy_enforced"] = self.enforced_cids
        stats["LazyStatic: Restored"] = self.n_restored
        stats["LazyStatic: Iterations"] = self.n_iterations

    @overrides
    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
        stats: LearningSolveStats,
        sample: Sample,
    ) -> None:
        assert solver.internal_solver is not None
        assert sample.after_load is not None
        assert sample.after_load.instance is not None

        logger.info("Predicting violated (static) lazy constraints...")
        if sample.after_load.instance.lazy_constraint_count == 0:
            logger.info("Instance does not have static lazy constraints. Skipping.")
        self.enforced_cids = set(self.sample_predict(sample))
        logger.info("Moving lazy constraints to the pool...")
        constraints = sample.after_load.constraints
        assert constraints is not None
        assert constraints.lazy is not None
        assert constraints.names is not None
        selected = tuple(
            (constraints.lazy[i] and constraints.names[i] not in self.enforced_cids)
            for i in range(len(constraints.lazy))
        )
        n_removed = sum(selected)
        n_kept = sum(constraints.lazy) - n_removed
        self.pool = constraints[selected]
        assert self.pool.names is not None
        solver.internal_solver.remove_constraints(self.pool.names)
        logger.info(f"{n_kept} lazy constraints kept; {n_removed} moved to the pool")
        stats["LazyStatic: Removed"] = n_removed
        stats["LazyStatic: Kept"] = n_kept
        stats["LazyStatic: Restored"] = 0
        self.n_restored = 0
        self.n_iterations = 0

    @overrides
    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        for c in y.keys():
            assert c in x
            self.classifiers[c] = self.classifier_prototype.clone()
            self.thresholds[c] = self.threshold_prototype.clone()
            self.classifiers[c].fit(x[c], y[c])
            self.thresholds[c].fit(self.classifiers[c], x[c], y[c])

    @overrides
    def iteration_cb(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
    ) -> bool:
        if solver.use_lazy_cb:
            return False
        else:
            return self._check_and_add(solver)

    @overrides
    def lazy_cb(
        self,
        solver: "LearningSolver",
        instance: "Instance",
        model: Any,
    ) -> None:
        self._check_and_add(solver)

    def sample_predict(self, sample: Sample) -> List[Hashable]:
        x, y, cids = self._sample_xy_with_cids(sample)
        enforced_cids: List[Hashable] = []
        for category in x.keys():
            if category not in self.classifiers:
                continue
            npx = np.array(x[category])
            proba = self.classifiers[category].predict_proba(npx)
            thr = self.thresholds[category].predict(npx)
            pred = list(proba[:, 1] > thr[1])
            for (i, is_selected) in enumerate(pred):
                if is_selected:
                    enforced_cids += [cids[category][i]]
        return enforced_cids

    @overrides
    def sample_xy(
        self,
        _: Optional[Instance],
        sample: Sample,
    ) -> Tuple[Dict[Hashable, List[List[float]]], Dict[Hashable, List[List[float]]]]:
        x, y, __ = self._sample_xy_with_cids(sample)
        return x, y

    def _check_and_add(self, solver: "LearningSolver") -> bool:
        assert solver.internal_solver is not None
        assert self.pool.names is not None
        if len(self.pool.names) == 0:
            logger.info("Lazy constraint pool is empty. Skipping violation check.")
            return False
        self.n_iterations += 1
        logger.info("Finding violated lazy constraints...")
        is_satisfied = solver.internal_solver.are_constraints_satisfied(
            self.pool,
            tol=self.violation_tolerance,
        )
        is_violated = tuple(not i for i in is_satisfied)
        violated_constraints = self.pool[is_violated]
        satisfied_constraints = self.pool[is_satisfied]
        self.pool = satisfied_constraints
        assert violated_constraints.names is not None
        assert satisfied_constraints.names is not None
        n_violated = len(violated_constraints.names)
        n_satisfied = len(satisfied_constraints.names)
        logger.info(f"Found {n_violated} violated lazy constraints found")
        if n_violated > 0:
            logger.info(
                "Enforcing {n_violated} lazy constraints; "
                f"{n_satisfied} left in the pool..."
            )
            solver.internal_solver.add_constraints(violated_constraints)
            for (i, name) in enumerate(violated_constraints.names):
                self.enforced_cids.add(name)
                self.n_restored += 1
            return True
        else:
            return False

    def _sample_xy_with_cids(
        self, sample: Sample
    ) -> Tuple[
        Dict[Hashable, List[List[float]]],
        Dict[Hashable, List[List[float]]],
        Dict[Hashable, List[str]],
    ]:
        x: Dict[Hashable, List[List[float]]] = {}
        y: Dict[Hashable, List[List[float]]] = {}
        cids: Dict[Hashable, List[str]] = {}
        assert sample.after_load is not None
        constraints = sample.after_load.constraints
        assert constraints is not None
        assert constraints.names is not None
        assert constraints.lazy is not None
        assert constraints.categories is not None
        for (cidx, cname) in enumerate(constraints.names):
            # Initialize categories
            if not constraints.lazy[cidx]:
                continue
            category = constraints.categories[cidx]
            if category is None:
                continue
            if category not in x:
                x[category] = []
                y[category] = []
                cids[category] = []

            # Features
            sf = sample.after_load
            if sample.after_lp is not None:
                sf = sample.after_lp
            assert sf.instance is not None
            assert sf.constraints is not None
            features = list(sf.instance.to_list())
            features.extend(sf.constraints.to_list(cidx))
            x[category].append(features)
            cids[category].append(cname)

            # Labels
            if (
                (sample.after_mip is not None)
                and (sample.after_mip.extra is not None)
                and ("lazy_enforced" in sample.after_mip.extra)
            ):
                if cname in sample.after_mip.extra["lazy_enforced"]:
                    y[category] += [[False, True]]
                else:
                    y[category] += [[True, False]]
        return x, y, cids
