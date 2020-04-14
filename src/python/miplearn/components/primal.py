#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from copy import deepcopy

from miplearn.classifiers.adaptive import AdaptiveClassifier
from sklearn.metrics import roc_curve

from .component import Component
from ..extractors import *

logger = logging.getLogger(__name__)


class PrimalSolutionComponent(Component):
    """
    A component that predicts primal solutions.
    """
    def __init__(self,
                 predictor=AdaptiveClassifier(),
                 mode="exact",
                 max_fpr=[1e-3, 1e-3],
                 min_threshold=[0.75, 0.75],
                 dynamic_thresholds=True,
                ):
        self.mode = mode
        self.predictors = {}
        self.is_warm_start_available = False
        self.max_fpr = max_fpr
        self.min_threshold = min_threshold
        self.thresholds = {}
        self.predictor_factory = predictor
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
                logger.debug("Fitting predictors[%s, %s]:" % (category, label))
                
                if callable(self.predictor_factory):
                    pred = self.predictor_factory(category, label)
                else:
                    pred = deepcopy(self.predictor_factory)
                self.predictors[category, label] = pred
                y = y_train[:, label].astype(int)
                pred.fit(x_train, y)

                # If y is either always one or always zero, set fixed threshold
                y_avg = np.average(y)
                if (not self.dynamic_thresholds) or y_avg <= 0.001 or y_avg >= 0.999:
                    self.thresholds[category, label] = self.min_threshold[label]
                    logger.debug("    Setting threshold to %.4f" % self.min_threshold[label])
                    continue
                
                # Calculate threshold dynamically using ROC curve
                y_scores = pred.predict_proba(x_train)[:, 1]
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
                    if (category, label) not in self.predictors.keys():
                        continue
                    ws = self.predictors[category, label].predict_proba(x_test[category])
                    logger.debug("%s[%s] ws=%.6f threshold=%.6f" %
                                 (var, index, ws[i, 1], self.thresholds[category, label]))
                    if ws[i, 1] >= self.thresholds[category, label]:
                        solution[var][index] = label
        return solution
