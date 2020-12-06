#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
import numpy as np

from copy import deepcopy

from tqdm import tqdm

from miplearn import Component
from miplearn.classifiers.counting import CountingClassifier
from miplearn.components import classifier_evaluation_dict
from miplearn.components.composite import CompositeComponent
from miplearn.components.lazy_static import LazyConstraint
from miplearn.extractors import InstanceIterator

logger = logging.getLogger(__name__)


class RelaxationComponent(Component):
    """
    A Component that tries to build a relaxation that is simultaneously strong and easy
    to solve.

    Currently, this component performs the following operations:
        - Drops all integrality constraints
        - Drops all inequality constraints that are not likely to be binding.

    In future versions of MIPLearn, this component may keep some integrality constraints
    and perform other operations.

    Parameters
    ----------
    classifier : Classifier, optional
        Classifier used to predict whether each constraint is binding or not. One deep
        copy of this classifier is made for each constraint category.
    threshold : float, optional
        If the probability that a constraint is binding exceeds this threshold, the
        constraint is dropped from the linear relaxation.
    slack_tolerance : float, optional
        If a constraint has slack greater than this threshold, then the constraint is
        considered loose. By default, this threshold equals a small positive number to
        compensate for numerical issues.
    check_dropped : bool, optional
        If `check_dropped` is true, then, after the problem is solved, the component
        verifies that all dropped constraints are still satisfied, re-adds the violated
        ones and resolves the problem. This loop continues until either no violations
        are found, or a maximum number of iterations is reached.
    violation_tolerance : float, optional
        If `check_dropped` is true, a constraint is considered satisfied during the
        check if its violation is smaller than this tolerance.
    max_iterations : int
        If `check_dropped` is true, set the maximum number of iterations in the lazy
        constraint loop.
    """

    def __init__(
        self,
        classifier=CountingClassifier(),
        threshold=0.95,
        slack_tolerance=1e-5,
        check_dropped=False,
        violation_tolerance=1e-5,
        max_iterations=3,
    ):
        self.steps = [
            RelaxIntegralityStep(),
            DropRedundantInequalitiesStep(
                classifier=classifier,
                threshold=threshold,
                slack_tolerance=slack_tolerance,
                violation_tolerance=violation_tolerance,
                max_iterations=max_iterations,
                check_dropped=check_dropped,
            ),
        ]
        self.composite = CompositeComponent(self.steps)

    def before_solve(self, solver, instance, model):
        self.composite.before_solve(solver, instance, model)

    def after_solve(self, solver, instance, model, results):
        self.composite.after_solve(solver, instance, model, results)

    def fit(self, training_instances):
        self.composite.fit(training_instances)

    def iteration_cb(self, solver, instance, model):
        return self.composite.iteration_cb(solver, instance, model)


class RelaxIntegralityStep(Component):
    def before_solve(self, solver, instance, _):
        logger.info("Relaxing integrality...")
        solver.internal_solver.relax()


class DropRedundantInequalitiesStep(Component):
    def __init__(
        self,
        classifier=CountingClassifier(),
        threshold=0.95,
        slack_tolerance=1e-5,
        check_dropped=False,
        violation_tolerance=1e-5,
        max_iterations=3,
    ):
        self.classifiers = {}
        self.classifier_prototype = classifier
        self.threshold = threshold
        self.slack_tolerance = slack_tolerance
        self.pool = []
        self.check_dropped = check_dropped
        self.violation_tolerance = violation_tolerance
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def before_solve(self, solver, instance, _):
        self.current_iteration = 0

        logger.info("Predicting redundant LP constraints...")
        cids = solver.internal_solver.get_constraint_ids()
        x, constraints = self.x(
            [instance],
            constraint_ids=cids,
            return_constraints=True,
        )
        y = self.predict(x)
        for category in y.keys():
            for i in range(len(y[category])):
                if y[category][i][0] == 1:
                    cid = constraints[category][i]
                    c = LazyConstraint(
                        cid=cid,
                        obj=solver.internal_solver.extract_constraint(cid),
                    )
                    self.pool += [c]
        logger.info("Extracted %d predicted constraints" % len(self.pool))

    def after_solve(self, solver, instance, model, results):
        instance.slacks = solver.internal_solver.get_constraint_slacks()

    def fit(self, training_instances):
        logger.debug("Extracting x and y...")
        x = self.x(training_instances)
        y = self.y(training_instances)
        logger.debug("Fitting...")
        for category in tqdm(x.keys(), desc="Fit (rlx:drop_ineq)"):
            if category not in self.classifiers:
                self.classifiers[category] = deepcopy(self.classifier_prototype)
            self.classifiers[category].fit(x[category], y[category])

    def x(self, instances, constraint_ids=None, return_constraints=False):
        x = {}
        constraints = {}
        for instance in tqdm(
            InstanceIterator(instances),
            desc="Extract (rlx:drop_ineq:x)",
            disable=len(instances) < 5,
        ):
            if constraint_ids is not None:
                cids = constraint_ids
            else:
                cids = instance.slacks.keys()
            for cid in cids:
                category = instance.get_constraint_category(cid)
                if category is None:
                    continue
                if category not in x:
                    x[category] = []
                    constraints[category] = []
                x[category] += [instance.get_constraint_features(cid)]
                constraints[category] += [cid]
        if return_constraints:
            return x, constraints
        else:
            return x

    def y(self, instances):
        y = {}
        for instance in tqdm(
            InstanceIterator(instances),
            desc="Extract (rlx:drop_ineq:y)",
            disable=len(instances) < 5,
        ):
            for (cid, slack) in instance.slacks.items():
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
            # x_cat = np.array(x_cat)
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
        if not self.check_dropped:
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
            logger.info(
                "%8d constraints %8d in the pool"
                % (len(constraints_to_add), len(self.pool))
            )
            return True
        else:
            return False
