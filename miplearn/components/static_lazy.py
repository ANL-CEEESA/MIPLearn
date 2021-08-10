#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Dict, Tuple, List, Any, TYPE_CHECKING, Set, Optional

import numpy as np
from overrides import overrides

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.threshold import MinProbabilityThreshold, Threshold
from miplearn.components.component import Component
from miplearn.features.sample import Sample
from miplearn.solvers.internal import Constraints
from miplearn.instance.base import Instance
from miplearn.types import LearningSolveStats, ConstraintName, ConstraintCategory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


class LazyConstraint:
    def __init__(self, cid: ConstraintName, obj: Any) -> None:
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
        self.classifiers: Dict[ConstraintCategory, Classifier] = {}
        self.thresholds: Dict[ConstraintCategory, Threshold] = {}
        self.pool: Constraints = Constraints()
        self.violation_tolerance: float = violation_tolerance
        self.enforced_cids: Set[ConstraintName] = set()
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
        sample.put_array(
            "mip_constr_lazy_enforced",
            np.array(list(self.enforced_cids), dtype="S"),
        )
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
        static_lazy_count = sample.get_scalar("static_constr_lazy_count")
        assert static_lazy_count is not None

        logger.info("Predicting violated (static) lazy constraints...")
        if static_lazy_count == 0:
            logger.info("Instance does not have static lazy constraints. Skipping.")
        self.enforced_cids = set(self.sample_predict(sample))
        logger.info("Moving lazy constraints to the pool...")
        constraints = Constraints.from_sample(sample)
        assert constraints.lazy is not None
        assert constraints.names is not None
        selected = [
            (constraints.lazy[i] and constraints.names[i] not in self.enforced_cids)
            for i in range(len(constraints.lazy))
        ]
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
        x: Dict[ConstraintCategory, np.ndarray],
        y: Dict[ConstraintCategory, np.ndarray],
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

    def sample_predict(self, sample: Sample) -> List[ConstraintName]:
        x, y, cids = self._sample_xy_with_cids(sample)
        enforced_cids: List[ConstraintName] = []
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
    ) -> Tuple[
        Dict[ConstraintCategory, List[List[float]]],
        Dict[ConstraintCategory, List[List[float]]],
    ]:
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
        is_violated = [not i for i in is_satisfied]
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
                f"Enforcing {n_violated} lazy constraints; "
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
        Dict[ConstraintCategory, List[List[float]]],
        Dict[ConstraintCategory, List[List[float]]],
        Dict[ConstraintCategory, List[ConstraintName]],
    ]:
        x: Dict[ConstraintCategory, List[List[float]]] = {}
        y: Dict[ConstraintCategory, List[List[float]]] = {}
        cids: Dict[ConstraintCategory, List[ConstraintName]] = {}
        instance_features = sample.get_array("static_instance_features")
        constr_features = sample.get_array("lp_constr_features")
        constr_names = sample.get_array("static_constr_names")
        constr_categories = sample.get_array("static_constr_categories")
        constr_lazy = sample.get_array("static_constr_lazy")
        lazy_enforced = sample.get_array("mip_constr_lazy_enforced")
        if constr_features is None:
            constr_features = sample.get_array("static_constr_features")

        assert instance_features is not None
        assert constr_features is not None
        assert constr_names is not None
        assert constr_categories is not None
        assert constr_lazy is not None

        for (cidx, cname) in enumerate(constr_names):
            # Initialize categories
            if not constr_lazy[cidx]:
                continue
            category = constr_categories[cidx]
            if len(category) == 0:
                continue
            if category not in x:
                x[category] = []
                y[category] = []
                cids[category] = []

            # Features
            features = list(instance_features)
            features.extend(constr_features[cidx])
            x[category].append(features)
            cids[category].append(cname)

            # Labels
            if lazy_enforced is not None:
                if cname in lazy_enforced:
                    y[category] += [[False, True]]
                else:
                    y[category] += [[True, False]]
        return x, y, cids
