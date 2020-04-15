#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from copy import deepcopy

from miplearn.classifiers.adaptive import AdaptiveClassifier
from miplearn.components import classifier_evaluation_dict
from sklearn.metrics import roc_curve

from .component import Component
from ..extractors import *

logger = logging.getLogger(__name__)


class PrimalSolutionComponent(Component):
    """
    A component that predicts primal solutions.
    """
    def __init__(self,
                 classifier=AdaptiveClassifier(),
                 mode="exact",
                 max_fpr=[1e-3, 1e-3],
                 min_threshold=[0.75, 0.75],
                 dynamic_thresholds=True,
                 ):
        self.mode = mode
        self.is_warm_start_available = False
        self.max_fpr = max_fpr
        self.min_threshold = min_threshold
        self.thresholds = {}
        self.classifiers = {}
        self.classifier_prototype = classifier
        self.dynamic_thresholds = dynamic_thresholds
    
    def before_solve(self, solver, instance, model):
        solution = self.predict(instance)
        if self.mode == "heuristic":
            solver.internal_solver.fix(solution)
        else:
            solver.internal_solver.set_warm_start(solution)
        
    def after_solve(self, solver, instance, model, results):
        pass
                
    def fit(self, training_instances):
        logger.debug("Extracting features...")
        features = VariableFeaturesExtractor().extract(training_instances)
        solutions = SolutionExtractor().extract(training_instances)
        
        for category in tqdm(features.keys(), desc="Fit (Primal)"):
            x_train = features[category]
            y_train = solutions[category]
            for label in [0, 1]:
                y = y_train[:, label].astype(int)

                logger.debug("Fitting predictors[%s, %s]:" % (category, label))
                if isinstance(self.classifier_prototype, list):
                    pred = deepcopy(self.classifier_prototype[label])
                else:
                    pred = deepcopy(self.classifier_prototype)
                pred.fit(x_train, y)
                self.classifiers[category, label] = pred

                # If y is either always one or always zero, set fixed threshold
                y_avg = np.average(y)
                if (not self.dynamic_thresholds) or y_avg <= 0.001 or y_avg >= 0.999:
                    self.thresholds[category, label] = self.min_threshold[label]
                    logger.debug("    Setting threshold to %.4f" % self.min_threshold[label])
                    continue
                
                proba = pred.predict_proba(x_train)
                assert isinstance(proba, np.ndarray), \
                    "classifier should return numpy array"
                assert proba.shape == (x_train.shape[0], 2),\
                    "classifier should return (%d,%d)-shaped array, not %s" % (
                        x_train.shape[0], 2, str(proba.shape))

                # Calculate threshold dynamically using ROC curve
                y_scores = proba[:, 1]
                fpr, tpr, thresholds = roc_curve(y, y_scores)
                k = 0
                while True:
                    if (k + 1) > len(fpr):
                        break
                    if fpr[k + 1] > self.max_fpr[label]:
                        break
                    if thresholds[k + 1] < self.min_threshold[label]:
                        break
                    k = k + 1
                logger.debug("    Setting threshold to %.4f (fpr=%.4f, tpr=%.4f)"%
                             (thresholds[k], fpr[k], tpr[k]))
                self.thresholds[category, label] = thresholds[k]
                
    def predict(self, instance):
        x_test = VariableFeaturesExtractor().extract([instance])
        solution = {}
        var_split = Extractor.split_variables(instance)
        for category in var_split.keys():
            for (i, (var, index)) in enumerate(var_split[category]):
                if var not in solution.keys():
                    solution[var] = {}
                solution[var][index] = None
                for label in [0, 1]:
                    if (category, label) not in self.classifiers.keys():
                        continue
                    ws = self.classifiers[category, label].predict_proba(x_test[category])
                    logger.debug("%s[%s] ws=%.6f threshold=%.6f" %
                                 (var, index, ws[i, 1], self.thresholds[category, label]))
                    if ws[i, 1] >= self.thresholds[category, label]:
                        solution[var][index] = label
        return solution

    def evaluate(self, instances):
        ev = {}
        for (instance_idx, instance) in enumerate(instances):
            solution_actual = instance.solution
            solution_pred = self.predict(instance)

            vars_all, vars_one, vars_zero = set(), set(), set()
            pred_one_positive, pred_zero_positive = set(), set()
            for (varname, var_dict) in solution_actual.items():
                for (idx, value) in var_dict.items():
                    vars_all.add((varname, idx))
                    if value > 0.5:
                        vars_one.add((varname, idx))
                    else:
                        vars_zero.add((varname, idx))
                    if solution_pred[varname][idx] is not None:
                        if solution_pred[varname][idx] > 0.5:
                            pred_one_positive.add((varname, idx))
                        else:
                            pred_zero_positive.add((varname, idx))
            pred_one_negative = vars_all - pred_one_positive
            pred_zero_negative = vars_all - pred_zero_positive

            tp_zero = len(pred_zero_positive & vars_zero)
            fp_zero = len(pred_zero_positive & vars_one)
            tn_zero = len(pred_zero_negative & vars_one)
            fn_zero = len(pred_zero_negative & vars_zero)

            tp_one = len(pred_one_positive & vars_one)
            fp_one = len(pred_one_positive & vars_zero)
            tn_one = len(pred_one_negative & vars_zero)
            fn_one = len(pred_one_negative & vars_one)

            ev[instance_idx] = {
                "Fix zero": classifier_evaluation_dict(tp_zero, tn_zero, fp_zero, fn_zero),
                "Fix one": classifier_evaluation_dict(tp_one, tn_one, fp_one, fn_one),
            }
        return ev
