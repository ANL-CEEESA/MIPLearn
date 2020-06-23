#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from copy import deepcopy

from .component import Component
from ..classifiers.adaptive import AdaptiveClassifier
from ..classifiers.threshold import MinPrecisionThreshold, DynamicThreshold
from ..components import classifier_evaluation_dict
from ..extractors import *

logger = logging.getLogger(__name__)


class PrimalSolutionComponent(Component):
    """
    A component that predicts primal solutions.
    """

    def __init__(self,
                 classifier=AdaptiveClassifier(),
                 mode="exact",
                 threshold=MinPrecisionThreshold(0.98)):
        self.mode = mode
        self.classifiers = {}
        self.thresholds = {}
        self.threshold_prototype = threshold
        self.classifier_prototype = classifier

    def before_solve(self, solver, instance, model):
        solution = self.predict(instance)
        if self.mode == "heuristic":
            solver.internal_solver.fix(solution)
        else:
            solver.internal_solver.set_warm_start(solution)

    def after_solve(self, solver, instance, model, results):
        pass

    def x(self, training_instances):
        return VariableFeaturesExtractor().extract(training_instances)

    def y(self, training_instances):
        return SolutionExtractor().extract(training_instances)

    def fit(self, training_instances, n_jobs=1):
        logger.debug("Extracting features...")
        features = VariableFeaturesExtractor().extract(training_instances)
        solutions = SolutionExtractor().extract(training_instances)

        for category in tqdm(features.keys(), desc="Fit (primal)"):
            x_train = features[category]
            for label in [0, 1]:
                y_train = solutions[category][:, label].astype(int)

                # If all samples are either positive or negative, make constant predictions
                y_avg = np.average(y_train)
                if y_avg < 0.001 or y_avg >= 0.999:
                    self.classifiers[category, label] = round(y_avg)
                    self.thresholds[category, label] = 0.50
                    continue

                # Create a copy of classifier prototype and train it
                if isinstance(self.classifier_prototype, list):
                    clf = deepcopy(self.classifier_prototype[label])
                else:
                    clf = deepcopy(self.classifier_prototype)
                clf.fit(x_train, y_train)

                # Find threshold (dynamic or static)
                if isinstance(self.threshold_prototype, DynamicThreshold):
                    self.thresholds[category, label] = self.threshold_prototype.find(clf, x_train, y_train)
                else:
                    self.thresholds[category, label] = deepcopy(self.threshold_prototype)

                self.classifiers[category, label] = clf

    def predict(self, instance):
        solution = {}
        x_test = VariableFeaturesExtractor().extract([instance])
        var_split = Extractor.split_variables(instance)
        for category in var_split.keys():
            n = len(var_split[category])
            for (i, (var, index)) in enumerate(var_split[category]):
                if var not in solution.keys():
                    solution[var] = {}
                solution[var][index] = None
            for label in [0, 1]:
                if (category, label) not in self.classifiers.keys():
                    continue
                clf = self.classifiers[category, label]
                if isinstance(clf, float) or isinstance(clf, int):
                    ws = np.array([[1 - clf, clf] for _ in range(n)])
                else:
                    ws = clf.predict_proba(x_test[category])
                assert ws.shape == (n, 2), "ws.shape should be (%d, 2) not %s" % (n, ws.shape)
                for (i, (var, index)) in enumerate(var_split[category]):
                    if ws[i, 1] >= self.thresholds[category, label]:
                        solution[var][index] = label
        return solution

    def evaluate(self, instances):
        ev = {"Fix zero": {},
              "Fix one": {}}
        for instance_idx in tqdm(range(len(instances)),
                                 desc="Evaluate (primal)"):
            instance = instances[instance_idx]
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

            ev["Fix zero"][instance_idx] = classifier_evaluation_dict(tp_zero, tn_zero, fp_zero, fn_zero)
            ev["Fix one"][instance_idx] = classifier_evaluation_dict(tp_one, tn_one, fp_one, fn_one)
        return ev
