#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from miplearn import Component
from miplearn.classifiers.counting import CountingClassifier
from miplearn.components import classifier_evaluation_dict
from miplearn.components.lazy_static import LazyConstraint
from miplearn.extractors import InstanceIterator

logger = logging.getLogger(__name__)


class DropRedundantInequalitiesStep(Component):
    """
    Component that predicts which inequalities are likely loose in the LP and removes
    them. Optionally, double checks after the problem is solved that all dropped
    inequalities were in fact redundant, and, if not, re-adds them to the problem.

    This component does not work on MIPs. All integrality constraints must be relaxed
    before this component is used.
    """

    def __init__(
        self,
        classifier=CountingClassifier(),
        threshold=0.95,
        slack_tolerance=1e-5,
        check_feasibility=False,
        violation_tolerance=1e-5,
        max_iterations=3,
    ):
        self.classifiers = {}
        self.classifier_prototype = classifier
        self.threshold = threshold
        self.slack_tolerance = slack_tolerance
        self.pool = []
        self.check_feasibility = check_feasibility
        self.violation_tolerance = violation_tolerance
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def before_solve(self, solver, instance, _):
        self.current_iteration = 0

        logger.info("Predicting redundant LP constraints...")
        x, constraints = self._x_test(
            instance,
            constraint_ids=solver.internal_solver.get_constraint_ids(),
        )
        y = self.predict(x)

        self.total_dropped = 0
        self.total_restored = 0
        self.total_kept = 0
        self.total_iterations = 0
        for category in y.keys():
            for i in range(len(y[category])):
                if y[category][i][0] == 1:
                    cid = constraints[category][i]
                    c = LazyConstraint(
                        cid=cid,
                        obj=solver.internal_solver.extract_constraint(cid),
                    )
                    self.pool += [c]
                    self.total_dropped += 1
                else:
                    self.total_kept += 1
        logger.info(f"Extracted {self.total_dropped} predicted constraints")

    def after_solve(
        self,
        solver,
        instance,
        model,
        stats,
        training_data,
    ):
        if "slacks" not in training_data.keys():
            training_data["slacks"] = solver.internal_solver.get_inequality_slacks()
        stats.update(
            {
                "DropRedundant: Kept": self.total_kept,
                "DropRedundant: Dropped": self.total_dropped,
                "DropRedundant: Restored": self.total_restored,
                "DropRedundant: Iterations": self.total_iterations,
            }
        )

    def fit(self, training_instances):
        logger.debug("Extracting x and y...")
        x = self.x(training_instances)
        y = self.y(training_instances)
        logger.debug("Fitting...")
        for category in tqdm(x.keys(), desc="Fit (rlx:drop_ineq)"):
            if category not in self.classifiers:
                self.classifiers[category] = deepcopy(self.classifier_prototype)
            self.classifiers[category].fit(x[category], y[category])

    @staticmethod
    def _x_test(instance, constraint_ids):
        x = {}
        constraints = {}
        cids = constraint_ids
        for cid in cids:
            category = instance.get_constraint_category(cid)
            if category is None:
                continue
            if category not in x:
                x[category] = []
                constraints[category] = []
            x[category] += [instance.get_constraint_features(cid)]
            constraints[category] += [cid]
        for category in x.keys():
            x[category] = np.array(x[category])
        return x, constraints

    @staticmethod
    def _x_train(instances):
        x = {}
        for instance in tqdm(
            InstanceIterator(instances),
            desc="Extract (rlx:drop_ineq:x)",
            disable=len(instances) < 5,
        ):
            for training_data in instance.training_data:
                cids = training_data["slacks"].keys()
                for cid in cids:
                    category = instance.get_constraint_category(cid)
                    if category is None:
                        continue
                    if category not in x:
                        x[category] = []
                    x[category] += [instance.get_constraint_features(cid)]
        for category in x.keys():
            x[category] = np.array(x[category])
        return x

    def x(self, instances):
        return self._x_train(instances)

    def y(self, instances):
        y = {}
        for instance in tqdm(
            InstanceIterator(instances),
            desc="Extract (rlx:drop_ineq:y)",
            disable=len(instances) < 5,
        ):
            for training_data in instance.training_data:
                for (cid, slack) in training_data["slacks"].items():
                    category = instance.get_constraint_category(cid)
                    if category is None:
                        continue
                    if category not in y:
                        y[category] = []
                    if slack > self.slack_tolerance:
                        y[category] += [[1]]
                    else:
                        y[category] += [[0]]
        return y

    def predict(self, x):
        y = {}
        for (category, x_cat) in x.items():
            if category not in self.classifiers:
                continue
            y[category] = []
            x_cat = np.array(x_cat)
            proba = self.classifiers[category].predict_proba(x_cat)
            for i in range(len(proba)):
                if proba[i][1] >= self.threshold:
                    y[category] += [[1]]
                else:
                    y[category] += [[0]]
        return y

    def evaluate(self, instance):
        x = self.x([instance])
        y_true = self.y([instance])
        y_pred = self.predict(x)
        tp, tn, fp, fn = 0, 0, 0, 0
        for category in y_true.keys():
            for i in range(len(y_true[category])):
                if y_pred[category][i][0] == 1:
                    if y_true[category][i][0] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if y_true[category][i][0] == 1:
                        fn += 1
                    else:
                        tn += 1
        return classifier_evaluation_dict(tp, tn, fp, fn)

    def iteration_cb(self, solver, instance, model):
        if not self.check_feasibility:
            return False
        if self.current_iteration >= self.max_iterations:
            return False
        self.current_iteration += 1
        logger.debug("Checking that dropped constraints are satisfied...")
        constraints_to_add = []
        for c in self.pool:
            if not solver.internal_solver.is_constraint_satisfied(
                c.obj,
                self.violation_tolerance,
            ):
                constraints_to_add.append(c)
        for c in constraints_to_add:
            self.pool.remove(c)
            solver.internal_solver.add_constraint(c.obj)
        if len(constraints_to_add) > 0:
            self.total_restored += len(constraints_to_add)
            logger.info(
                "%8d constraints %8d in the pool"
                % (len(constraints_to_add), len(self.pool))
            )
            self.total_iterations += 1
            return True
        else:
            return False
