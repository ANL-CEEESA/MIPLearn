#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from copy import deepcopy

from miplearn.classifiers.counting import CountingClassifier

from .component import Component
from ..extractors import *

logger = logging.getLogger(__name__)


class LazyConstraintsComponent(Component):
    """
    A component that predicts which lazy constraints to enforce.
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
        logger.info("Predicting violated lazy constraints...")
        violations = []
        features = InstanceFeaturesExtractor().extract([instance])
        for (v, classifier) in self.classifiers.items():
            proba = classifier.predict_proba(features)
            if proba[0][1] > self.threshold:
                violations += [v]

        logger.info("Enforcing %d constraints..." % len(violations))
        for v in violations:
            cut = instance.build_lazy_constraint(model, v)
            solver.internal_solver.add_constraint(cut)

    def after_solve(self, solver, instance, model, results):
        pass
                
    def fit(self, training_instances):
        logger.debug("Fitting...")
        features = InstanceFeaturesExtractor().extract(training_instances)

        self.classifiers = {}
        violation_to_instance_idx = {}
        for (idx, instance) in enumerate(training_instances):
            for v in instance.found_violations:
                if v not in self.classifiers:
                    self.classifiers[v] = deepcopy(self.classifier_prototype)
                    violation_to_instance_idx[v] = []
                violation_to_instance_idx[v] += [idx]

        for (v, classifier) in self.classifiers.items():
            logger.debug("Training: %s" % (str(v)))
            label = np.zeros(len(training_instances))
            label[violation_to_instance_idx[v]] = 1.0
            classifier.fit(features, label)

    def predict(self, instance, model=None):
        return self.violations
