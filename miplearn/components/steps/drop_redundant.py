#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy

import numpy as np
from p_tqdm import p_umap
from tqdm import tqdm

from miplearn.classifiers.counting import CountingClassifier
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.components.static_lazy import LazyConstraint

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
        check_feasibility=True,
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
        self.n_iterations = 0
        self.n_restored = 0

    def before_solve_mip(
        self,
        solver,
        instance,
        model,
        stats,
        features,
        training_data,
    ):
        self.n_iterations = 0
        self.n_restored = 0
        self.current_iteration = 0

        logger.info("Predicting redundant LP constraints...")
        x, constraints = self.x(
            instance,
            constraint_ids=solver.internal_solver.get_constraint_ids(),
        )
        y = self.predict(x)

        self.pool = []
        n_dropped = 0
        n_kept = 0
        for category in y.keys():
            for i in range(len(y[category])):
                if y[category][i][1] == 1:
                    cid = constraints[category][i]
                    c = LazyConstraint(
                        cid=cid,
                        obj=solver.internal_solver.extract_constraint(cid),
                    )
                    self.pool += [c]
                    n_dropped += 1
                else:
                    n_kept += 1
        stats["DropRedundant: Kept"] = n_kept
        stats["DropRedundant: Dropped"] = n_dropped
        logger.info(f"Extracted {n_dropped} predicted constraints")

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
        stats["DropRedundant: Iterations"] = self.n_iterations
        stats["DropRedundant: Restored"] = self.n_restored

    def fit(self, training_instances, n_jobs=1):
        x, y = self.x_y(training_instances, n_jobs=n_jobs)
        for category in tqdm(x.keys(), desc="Fit (drop)"):
            if category not in self.classifiers:
                self.classifiers[category] = deepcopy(self.classifier_prototype)
            self.classifiers[category].fit(x[category], np.array(y[category]))

    @staticmethod
    def x(instance, constraint_ids):
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

    def x_y(self, instances, n_jobs=1):
        def _extract(instance):
            x = {}
            y = {}
            for training_data in instance.training_data:
                for (cid, slack) in training_data.slacks.items():
                    category = instance.get_constraint_category(cid)
                    if category is None:
                        continue
                    if category not in x:
                        x[category] = []
                    if category not in y:
                        y[category] = []
                    if slack > self.slack_tolerance:
                        y[category] += [[False, True]]
                    else:
                        y[category] += [[True, False]]
                    x[category] += [instance.get_constraint_features(cid)]
            return x, y

        if n_jobs == 1:
            results = [_extract(i) for i in tqdm(instances, desc="Extract (drop 1/3)")]
        else:
            results = p_umap(
                _extract,
                instances,
                num_cpus=n_jobs,
                desc="Extract (drop 1/3)",
            )

        x_combined = {}
        y_combined = {}
        for (x, y) in tqdm(results, desc="Extract (drop 2/3)"):
            for category in x.keys():
                if category not in x_combined:
                    x_combined[category] = []
                    y_combined[category] = []
                x_combined[category] += x[category]
                y_combined[category] += y[category]

        for category in tqdm(x_combined.keys(), desc="Extract (drop 3/3)"):
            x_combined[category] = np.array(x_combined[category])
            y_combined[category] = np.array(y_combined[category])

        return x_combined, y_combined

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
                    y[category] += [[False, True]]
                else:
                    y[category] += [[True, False]]
        return y

    def evaluate(self, instance, n_jobs=1):
        x, y_true = self.x_y([instance], n_jobs=n_jobs)
        y_pred = self.predict(x)
        tp, tn, fp, fn = 0, 0, 0, 0
        for category in tqdm(
            y_true.keys(),
            disable=len(y_true) < 100,
            desc="Eval (drop)",
        ):
            for i in range(len(y_true[category])):
                if (category in y_pred) and (y_pred[category][i][1] == 1):
                    if y_true[category][i][1] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if y_true[category][i][1] == 1:
                        fn += 1
                    else:
                        tn += 1
        return classifier_evaluation_dict(tp, tn, fp, fn)

    def iteration_cb(self, solver, instance, model):
        if not self.check_feasibility:
            return False
        if self.current_iteration >= self.max_iterations:
            return False
        if solver.internal_solver.is_infeasible():
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
            self.n_restored += len(constraints_to_add)
            logger.info(
                "%8d constraints %8d in the pool"
                % (len(constraints_to_add), len(self.pool))
            )
            self.n_iterations += 1
            return True
        else:
            return False
