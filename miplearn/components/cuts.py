#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import sys
from copy import deepcopy

from miplearn.classifiers.counting import CountingClassifier
from miplearn.components import classifier_evaluation_dict

from .component import Component
from ..extractors import *

logger = logging.getLogger(__name__)


class UserCutsComponent(Component):
    """
    A component that predicts which user cuts to enforce.
    """
    
    def __init__(self,
                 classifier=CountingClassifier(),
                 threshold=0.05):
        self.violations = set()
        self.count = {}
        self.n_samples = 0
        self.threshold = threshold
        self.classifier_prototype = classifier
        self.classifiers = {}

    def before_solve(self, solver, instance, model):
        logger.info("Predicting violated user cuts...")
        violations = self.predict(instance)
        logger.info("Enforcing %d cuts..." % len(violations))
        for v in violations:
            cut = instance.build_user_cut(model, v)
            solver.internal_solver.add_constraint(cut)

    def after_solve(self, solver, instance, model, results):
        pass
                
    def fit(self, training_instances):
        logger.debug("Fitting...")
        features = InstanceFeaturesExtractor().extract(training_instances)

        self.classifiers = {}
        violation_to_instance_idx = {}
        for (idx, instance) in enumerate(training_instances):
            for v in instance.found_violated_user_cuts:
                if v not in self.classifiers:
                    self.classifiers[v] = deepcopy(self.classifier_prototype)
                    violation_to_instance_idx[v] = []
                violation_to_instance_idx[v] += [idx]

        for (v, classifier) in tqdm(self.classifiers.items(),
                                    desc="Fit (user cuts)",
                                    disable=not sys.stdout.isatty(),
                                   ):
            logger.debug("Training: %s" % (str(v)))
            label = np.zeros(len(training_instances))
            label[violation_to_instance_idx[v]] = 1.0
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
            all_violations |= set(instance.found_violated_user_cuts)
        for idx in tqdm(range(len(instances)),
                        desc="Evaluate (lazy)",
                        disable=not sys.stdout.isatty(),
                       ):
            instance = instances[idx]
            condition_positive = set(instance.found_violated_user_cuts)
            condition_negative = all_violations - condition_positive
            pred_positive = set(self.predict(instance)) & all_violations
            pred_negative = all_violations - pred_positive
            tp = len(pred_positive & condition_positive)
            tn = len(pred_negative & condition_negative)
            fp = len(pred_positive & condition_negative)
            fn = len(pred_negative & condition_positive)
            results[idx] = classifier_evaluation_dict(tp, tn, fp, fn)
        return results
