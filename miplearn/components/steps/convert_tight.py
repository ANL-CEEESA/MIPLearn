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
from miplearn.extractors import InstanceIterator

logger = logging.getLogger(__name__)


class ConvertTightIneqsIntoEqsStep(Component):
    """
    Component that predicts which inequality constraints are likely to be binding in
    the LP relaxation of the problem and converts them into equality constraints.
    Optionally double checks that the conversion process did not affect feasibility
    or optimality of the problem.

    This component does not work on MIPs. All integrality constraints must be relaxed
    before this component is used.
    """

    def __init__(
        self,
        classifier=CountingClassifier(),
        threshold=0.95,
        slack_tolerance=0.0,
    ):
        self.classifiers = {}
        self.classifier_prototype = classifier
        self.threshold = threshold
        self.slack_tolerance = slack_tolerance

    def before_solve(self, solver, instance, _):
        logger.info("Predicting tight LP constraints...")
        cids = solver.internal_solver.get_constraint_ids()
        x, constraints = self.x(
            [instance],
            constraint_ids=cids,
            return_constraints=True,
        )
        y = self.predict(x)
        n_converted = 0
        for category in y.keys():
            for i in range(len(y[category])):
                if y[category][i][0] == 1:
                    cid = constraints[category][i]
                    solver.internal_solver.set_constraint_sense(cid, "=")
                    n_converted += 1
        logger.info(f"Converted {n_converted} inequalities into equalities")

    def after_solve(self, solver, instance, model, results):
        instance.slacks = solver.internal_solver.get_constraint_slacks()

    def fit(self, training_instances):
        logger.debug("Extracting x and y...")
        x = self.x(training_instances)
        y = self.y(training_instances)
        logger.debug("Fitting...")
        for category in tqdm(x.keys(), desc="Fit (rlx:conv_ineqs)"):
            if category not in self.classifiers:
                self.classifiers[category] = deepcopy(self.classifier_prototype)
            self.classifiers[category].fit(x[category], y[category])

    def x(
        self,
        instances,
        constraint_ids=None,
        return_constraints=False,
    ):
        x = {}
        constraints = {}
        for instance in tqdm(
            InstanceIterator(instances),
            desc="Extract (rlx:conv_ineqs:x)",
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
            desc="Extract (rlx:conv_ineqs:y)",
            disable=len(instances) < 5,
        ):
            for (cid, slack) in instance.slacks.items():
                category = instance.get_constraint_category(cid)
                if category is None:
                    continue
                if category not in y:
                    y[category] = []
                if 0 <= slack <= self.slack_tolerance:
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
