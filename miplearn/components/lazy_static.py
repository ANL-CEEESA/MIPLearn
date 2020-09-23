#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import sys
from copy import deepcopy

from miplearn.classifiers.counting import CountingClassifier
from .component import Component
from ..extractors import *

logger = logging.getLogger(__name__)


class LazyConstraint:
    def __init__(self, cid, obj):
        self.cid = cid
        self.obj = obj


class StaticLazyConstraintsComponent(Component):
    def __init__(self,
                 classifier=CountingClassifier(),
                 threshold=0.05):
        self.threshold = threshold
        self.classifier_prototype = classifier
        self.classifiers = {}
        self.pool = []

    def before_solve(self, solver, instance, model):
        instance.found_violated_lazy_constraints = []
        if instance.has_static_lazy_constraints():
            self._extract_and_predict_static(solver, instance)

    def after_solve(self, solver, instance, model, results):
        pass

    def after_iteration(self, solver, instance, model):
        logger.debug("Finding violated (static) lazy constraints...")
        n_added = 0
        for c in self.pool:
            if not solver.internal_solver.is_constraint_satisfied(c.obj):
                self.pool.remove(c)
                solver.internal_solver.add_constraint(c.obj)
                instance.found_violated_lazy_constraints += [c.cid]
                n_added += 1
        if n_added > 0:
            logger.debug("    %d violations found" % n_added)
            return True
        else:
            return False
                
    def fit(self, training_instances):
        logger.debug("Extracting x and y...")
        x = self.x(training_instances)
        y = self.y(training_instances)

        logger.debug("Fitting...")
        for category in tqdm(x.keys(),
                             desc="Fit (lazy)",
                             disable=not sys.stdout.isatty()):
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
        for cid in solver.internal_solver.get_constraint_names():
            if instance.is_constraint_lazy(cid):
                category = instance.get_lazy_constraint_category(cid)
                if category not in self.classifiers:
                    continue
                if category not in x:
                    x[category] = []
                    constraints[category] = []
                x[category] += [instance.get_lazy_constraint_features(cid)]
                c = LazyConstraint(cid=cid,
                                   obj=solver.internal_solver.extract_constraint(cid))
                constraints[category] += [c]
                self.pool.append(c)
        for (category, x_values) in x.items():
            if isinstance(x_values[0], np.ndarray):
                x[category] = np.array(x_values)
            proba = self.classifiers[category].predict_proba(x[category])
            for i in range(len(proba)):
                if proba[i][1] > self.threshold:
                    c = constraints[category][i]
                    self.pool.remove(c)
                    solver.internal_solver.add_constraint(c.obj)
                    instance.found_violated_lazy_constraints += [c.cid]

    def _collect_constraints(self, train_instances):
        constraints = {}
        for instance in train_instances:
            for cid in instance.found_violated_lazy_constraints:
                category = instance.get_lazy_constraint_category(cid)
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
                    result[category].append(instance.get_lazy_constraint_features(cid))
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
