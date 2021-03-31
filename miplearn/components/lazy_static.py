#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
from tqdm.auto import tqdm

from miplearn.classifiers.counting import CountingClassifier
from miplearn.components.component import Component
from miplearn.types import TrainingSample

logger = logging.getLogger(__name__)


class LazyConstraint:
    def __init__(self, cid, obj):
        self.cid = cid
        self.obj = obj


class StaticLazyConstraintsComponent(Component):
    def __init__(
        self,
        classifier=CountingClassifier(),
        threshold=0.05,
        use_two_phase_gap=True,
        large_gap=1e-2,
        violation_tolerance=-0.5,
    ):
        self.threshold = threshold
        self.classifier_prototype = classifier
        self.classifiers = {}
        self.pool = []
        self.original_gap = None
        self.large_gap = large_gap
        self.is_gap_large = False
        self.use_two_phase_gap = use_two_phase_gap
        self.violation_tolerance = violation_tolerance

    def before_solve_mip(self, solver, instance, model):
        self.pool = []
        if not solver.use_lazy_cb and self.use_two_phase_gap:
            logger.info("Increasing gap tolerance to %f", self.large_gap)
            self.original_gap = solver.gap_tolerance
            self.is_gap_large = True
            solver.internal_solver.set_gap_tolerance(self.large_gap)

        instance.found_violated_lazy_constraints = []
        if instance.has_static_lazy_constraints():
            self._extract_and_predict_static(solver, instance)

    def after_solve_mip(
        self,
        solver,
        instance,
        model,
        stats,
        training_data,
    ):
        pass

    def iteration_cb(self, solver, instance, model):
        if solver.use_lazy_cb:
            return False
        else:
            should_repeat = self._check_and_add(instance, solver)
            if should_repeat:
                return True
            else:
                if self.is_gap_large:
                    logger.info("Restoring gap tolerance to %f", self.original_gap)
                    solver.internal_solver.set_gap_tolerance(self.original_gap)
                    self.is_gap_large = False
                    return True
                else:
                    return False

    def lazy_cb(self, solver, instance, model):
        self._check_and_add(instance, solver)

    def _check_and_add(self, instance, solver):
        logger.debug("Finding violated lazy constraints...")
        constraints_to_add = []
        for c in self.pool:
            if not solver.internal_solver.is_constraint_satisfied(
                c.obj, tol=self.violation_tolerance
            ):
                constraints_to_add.append(c)
        for c in constraints_to_add:
            self.pool.remove(c)
            solver.internal_solver.add_constraint(c.obj)
            instance.found_violated_lazy_constraints += [c.cid]
        if len(constraints_to_add) > 0:
            logger.info(
                "%8d lazy constraints added %8d in the pool"
                % (len(constraints_to_add), len(self.pool))
            )
            return True
        else:
            return False

    def fit(self, training_instances):
        training_instances = [
            t
            for t in training_instances
            if hasattr(t, "found_violated_lazy_constraints")
        ]

        logger.debug("Extracting x and y...")
        x = self.x(training_instances)
        y = self.y(training_instances)

        logger.debug("Fitting...")
        for category in tqdm(
            x.keys(), desc="Fit (lazy)", disable=not sys.stdout.isatty()
        ):
            if category not in self.classifiers:
                self.classifiers[category] = deepcopy(self.classifier_prototype)
            self.classifiers[category].fit(x[category], y[category])

    def predict(self, instance):
        pass

    def evaluate(self, instances):
        pass

    def _extract_and_predict_static(self, solver, instance):
        x = {}
        constraints = {}
        logger.info("Extracting lazy constraints...")
        for cid in solver.internal_solver.get_constraint_ids():
            if instance.is_constraint_lazy(cid):
                category = instance.get_constraint_category(cid)
                if category not in x:
                    x[category] = []
                    constraints[category] = []
                x[category] += [instance.get_constraint_features(cid)]
                c = LazyConstraint(
                    cid=cid,
                    obj=solver.internal_solver.extract_constraint(cid),
                )
                constraints[category] += [c]
                self.pool.append(c)
        logger.info("%8d lazy constraints extracted" % len(self.pool))
        logger.info("Predicting required lazy constraints...")
        n_added = 0
        for (category, x_values) in x.items():
            if category not in self.classifiers:
                continue
            if isinstance(x_values[0], np.ndarray):
                x[category] = np.array(x_values)
            proba = self.classifiers[category].predict_proba(x[category])
            for i in range(len(proba)):
                if proba[i][1] > self.threshold:
                    n_added += 1
                    c = constraints[category][i]
                    self.pool.remove(c)
                    solver.internal_solver.add_constraint(c.obj)
                    instance.found_violated_lazy_constraints += [c.cid]
        logger.info(
            "%8d lazy constraints added %8d in the pool"
            % (
                n_added,
                len(self.pool),
            )
        )

    def _collect_constraints(self, train_instances):
        constraints = {}
        for instance in train_instances:
            for cid in instance.found_violated_lazy_constraints:
                category = instance.get_constraint_category(cid)
                if category not in constraints:
                    constraints[category] = set()
                constraints[category].add(cid)
        for (category, cids) in constraints.items():
            constraints[category] = sorted(list(cids))
        return constraints

    def x(self, train_instances):
        result = {}
        constraints = self._collect_constraints(train_instances)
        for (category, cids) in constraints.items():
            result[category] = []
            for instance in train_instances:
                for cid in cids:
                    result[category].append(instance.get_constraint_features(cid))
        return result

    def y(self, train_instances):
        result = {}
        constraints = self._collect_constraints(train_instances)
        for (category, cids) in constraints.items():
            result[category] = []
            for instance in train_instances:
                for cid in cids:
                    if cid in instance.found_violated_lazy_constraints:
                        result[category].append([0, 1])
                    else:
                        result[category].append([1, 0])
        return result

    @staticmethod
    def xy_sample(
        instance: Any,
        sample: TrainingSample,
    ) -> Tuple[Dict, Dict]:
        x: Dict = {}
        y: Dict = {}
        if "LazyStatic: All" not in sample:
            return x, y
        for cid in sorted(sample["LazyStatic: All"]):
            cfeatures = instance.features["Constraints"][cid]
            category = cfeatures["Category"]
            if category is None:
                continue
            if category not in x:
                x[category] = []
                y[category] = []
            x[category] += [cfeatures["User features"]]
            if cid in sample["LazyStatic: Enforced"]:
                y[category] += [[False, True]]
            else:
                y[category] += [[True, False]]
        return x, y
