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
        violations = self.predict(instance)
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

        for (v, classifier) in tqdm(self.classifiers.items(), desc="Fit (lazy)"):
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
        
        def _classifier_evaluation_dict(tp, tn, fp, fn):
            p = tp + fn
            n = fp + tn
            d = {
                "Predicted positive": fp + tp,
                "Predicted negative": fn + tn,
                "Condition positive": p,
                "Condition negative": n,
                "True positive": tp,
                "True negative": tn,
                "False positive": fp,
                "False negative": fn,
            }
            d["Accuracy"] = (tp + tn) / (p + n)
            d["F1 score"] = (2 * tp) / (2 * tp + fp + fn)
            d["Recall"] = tp / p
            d["Precision"] = tp / (tp + fp)
            T = (p + n) / 100.0
            d["Predicted positive (%)"] = d["Predicted positive"] / T
            d["Predicted negative (%)"] = d["Predicted negative"] / T
            d["Condition positive (%)"] = d["Condition positive"] / T
            d["Condition negative (%)"] = d["Condition negative"] / T
            d["True positive (%)"] = d["True positive"] / T
            d["True negative (%)"] = d["True negative"] / T
            d["False positive (%)"] = d["False positive"] / T
            d["False negative (%)"] = d["False negative"] / T
            return d
        
        results = {}
        
        all_violations = set()
        for instance in instances:
            all_violations |= set(instance.found_violations)
            
        for idx in tqdm(range(len(instances)), desc="Evaluate (lazy)"):
            instance = instances[idx]
            condition_positive = set(instance.found_violations)
            condition_negative = all_violations - condition_positive
            pred_positive = set(self.predict(instance)) & all_violations
            pred_negative = all_violations - pred_positive
            
            tp = len(pred_positive & condition_positive)
            tn = len(pred_negative & condition_negative)
            fp = len(pred_positive & condition_negative)
            fn = len(pred_negative & condition_positive)
            
            results[idx] = _classifier_evaluation_dict(tp, tn, fp, fn)
            
            
        return results