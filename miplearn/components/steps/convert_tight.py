#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from miplearn.classifiers.counting import CountingClassifier
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.components.steps.drop_redundant import DropRedundantInequalitiesStep

logger = logging.getLogger(__name__)


class ConvertTightIneqsIntoEqsStep(Component):
    """
    Component that predicts which inequality constraints are likely to be binding in
    the LP relaxation of the problem and converts them into equality constraints.

    This component always makes sure that the conversion process does not affect the
    feasibility of the problem. It can also, optionally, make sure that it does not affect
    the optimality, but this may be expensive.

    This component does not work on MIPs. All integrality constraints must be relaxed
    before this component is used.
    """

    def __init__(
        self,
        classifier=CountingClassifier(),
        threshold=0.95,
        slack_tolerance=0.0,
        check_optimality=False,
    ):
        self.classifiers = {}
        self.classifier_prototype = classifier
        self.threshold = threshold
        self.slack_tolerance = slack_tolerance
        self.check_optimality = check_optimality
        self.converted = []
        self.original_sense = {}
        self.n_restored = 0
        self.n_infeasible_iterations = 0
        self.n_suboptimal_iterations = 0

    def before_solve_mip(
        self,
        solver,
        instance,
        model,
        stats,
        features,
        training_data,
    ):
        self.n_restored = 0
        self.n_infeasible_iterations = 0
        self.n_suboptimal_iterations = 0

        logger.info("Predicting tight LP constraints...")
        x, constraints = DropRedundantInequalitiesStep.x(
            instance,
            constraint_ids=solver.internal_solver.get_constraint_ids(),
        )
        y = self.predict(x)

        n_converted = 0
        n_kept = 0
        for category in y.keys():
            for i in range(len(y[category])):
                if y[category][i][0] == 1:
                    cid = constraints[category][i]
                    s = solver.internal_solver.get_constraint_sense(cid)
                    self.original_sense[cid] = s
                    solver.internal_solver.set_constraint_sense(cid, "=")
                    self.converted += [cid]
                    n_converted += 1
                else:
                    n_kept += 1
        stats["ConvertTight: Kept"] = n_kept
        stats["ConvertTight: Converted"] = n_converted

        logger.info(f"Converted {n_converted} inequalities")

    def after_solve_mip(
        self,
        solver,
        instance,
        model,
        stats,
        features,
        training_data,
    ):
        if training_data.slacks is None:
            training_data.slacks = solver.internal_solver.get_inequality_slacks()
        stats["ConvertTight: Restored"] = self.n_restored
        stats["ConvertTight: Inf iterations"] = self.n_infeasible_iterations
        stats["ConvertTight: Subopt iterations"] = self.n_suboptimal_iterations

    def fit(self, training_instances):
        logger.debug("Extracting x and y...")
        x = self.x(training_instances)
        y = self.y(training_instances)
        logger.debug("Fitting...")
        for category in tqdm(x.keys(), desc="Fit (rlx:conv_ineqs)"):
            if category not in self.classifiers:
                self.classifiers[category] = deepcopy(self.classifier_prototype)
            self.classifiers[category].fit(x[category], y[category])

    @staticmethod
    def _x_train(instances):
        x = {}
        for instance in tqdm(
            instances,
            desc="Extract (drop:x)",
            disable=len(instances) < 5,
        ):
            for training_data in instance.training_data:
                cids = training_data.slacks.keys()
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
            instances,
            desc="Extract (rlx:conv_ineqs:y)",
            disable=len(instances) < 5,
        ):
            for (cid, slack) in instance.training_data[0].slacks.items():
                category = instance.get_constraint_category(cid)
                if category is None:
                    continue
                if category not in y:
                    y[category] = []
                if 0 <= slack <= self.slack_tolerance:
                    y[category] += [[False, True]]
                else:
                    y[category] += [[True, False]]
            for category in y.keys():
                y[category] = np.array(y[category], dtype=np.bool8)
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
        is_infeasible, is_suboptimal = False, False
        restored = []

        def check_pi(msense, csense, pi):
            if csense == "=":
                return True
            if msense == "max":
                if csense == "<":
                    return pi >= 0
                else:
                    return pi <= 0
            else:
                if csense == ">":
                    return pi >= 0
                else:
                    return pi <= 0

        def restore(cid):
            nonlocal restored
            csense = self.original_sense[cid]
            solver.internal_solver.set_constraint_sense(cid, csense)
            restored += [cid]

        if solver.internal_solver.is_infeasible():
            for cid in self.converted:
                pi = solver.internal_solver.get_dual(cid)
                if abs(pi) > 0:
                    is_infeasible = True
                    restore(cid)
        elif self.check_optimality:
            random.shuffle(self.converted)
            n_restored = 0
            for cid in self.converted:
                if n_restored >= 100:
                    break
                pi = solver.internal_solver.get_dual(cid)
                csense = self.original_sense[cid]
                msense = solver.internal_solver.get_sense()
                if not check_pi(msense, csense, pi):
                    is_suboptimal = True
                    restore(cid)
                    n_restored += 1

        for cid in restored:
            self.converted.remove(cid)

        if len(restored) > 0:
            self.n_restored += len(restored)
            if is_infeasible:
                self.n_infeasible_iterations += 1
            if is_suboptimal:
                self.n_suboptimal_iterations += 1
            logger.info(f"Restored {len(restored)} inequalities")
            return True
        else:
            return False
