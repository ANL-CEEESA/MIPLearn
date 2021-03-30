#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
from copy import deepcopy
from typing import Any, Dict

import numpy as np
from tqdm.auto import tqdm

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.extractors import InstanceFeaturesExtractor, InstanceIterator

logger = logging.getLogger(__name__)


class DynamicLazyConstraintsComponent(Component):
    """
    A component that predicts which lazy constraints to enforce.
    """

    def __init__(
        self,
        classifier: Classifier = CountingClassifier(),
        threshold: float = 0.05,
    ):
        self.threshold: float = threshold
        self.classifier_prototype: Classifier = classifier
        self.classifiers: Dict[Any, Classifier] = {}

    def before_solve_mip(self, solver, instance, model):
        instance.found_violated_lazy_constraints = []
        logger.info("Predicting violated lazy constraints...")
        violations = self.predict(instance)
        logger.info("Enforcing %d lazy constraints..." % len(violations))
        for v in violations:
            cut = instance.build_lazy_constraint(model, v)
            solver.internal_solver.add_constraint(cut)

    def iteration_cb(self, solver, instance, model):
        logger.debug("Finding violated (dynamic) lazy constraints...")
        violations = instance.find_violated_lazy_constraints(model)
        if len(violations) == 0:
            return False
        instance.found_violated_lazy_constraints += violations
        logger.debug("    %d violations found" % len(violations))
        for v in violations:
            cut = instance.build_lazy_constraint(model, v)
            solver.internal_solver.add_constraint(cut)
        return True

    def after_solve_mip(
        self,
        solver,
        instance,
        model,
        stats,
        training_data,
    ):
        pass

    def fit(self, training_instances):
        logger.debug("Fitting...")
        features = InstanceFeaturesExtractor().extract(training_instances)

        self.classifiers = {}
        violation_to_instance_idx = {}
        for (idx, instance) in enumerate(InstanceIterator(training_instances)):
            for v in instance.found_violated_lazy_constraints:
                if isinstance(v, list):
                    v = tuple(v)
                if v not in self.classifiers:
                    self.classifiers[v] = deepcopy(self.classifier_prototype)
                    violation_to_instance_idx[v] = []
                violation_to_instance_idx[v] += [idx]

        for (v, classifier) in tqdm(
            self.classifiers.items(),
            desc="Fit (lazy)",
            disable=not sys.stdout.isatty(),
        ):
            logger.debug("Training: %s" % (str(v)))
            label = [[True, False] for i in training_instances]
            for idx in violation_to_instance_idx[v]:
                label[idx] = [False, True]
            label = np.array(label, dtype=np.bool8)
            classifier.fit(features, label)

    def predict(self, instance):
        violations = []
        features = InstanceFeaturesExtractor().extract([instance])
        for (v, classifier) in self.classifiers.items():
            proba = classifier.predict_proba(features)
            if proba[0][1] > self.threshold:
                violations += [v]
        return violations

    def evaluate(self, instances):
        results = {}
        all_violations = set()
        for instance in instances:
            all_violations |= set(instance.found_violated_lazy_constraints)
        for idx in tqdm(
            range(len(instances)),
            desc="Evaluate (lazy)",
            disable=not sys.stdout.isatty(),
        ):
            instance = instances[idx]
            condition_positive = set(instance.found_violated_lazy_constraints)
            condition_negative = all_violations - condition_positive
            pred_positive = set(self.predict(instance)) & all_violations
            pred_negative = all_violations - pred_positive
            tp = len(pred_positive & condition_positive)
            tn = len(pred_negative & condition_negative)
            fp = len(pred_positive & condition_negative)
            fn = len(pred_negative & condition_positive)
            results[idx] = classifier_evaluation_dict(tp, tn, fp, fn)
        return results
