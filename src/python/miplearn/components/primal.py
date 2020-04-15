#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from copy import deepcopy

from miplearn.classifiers.adaptive import AdaptiveClassifier
from miplearn.components import classifier_evaluation_dict
from sklearn.metrics import roc_curve
from p_tqdm import p_map

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

    def fit(self, training_instances, n_jobs=1):
        logger.debug("Extracting features...")
        features = VariableFeaturesExtractor().extract(training_instances)
        solutions = SolutionExtractor().extract(training_instances)

        def _fit(args):
            category, label = args[0], args[1]
            x_train = features[category]
            y_train = solutions[category]
            y = y_train[:, label].astype(int)

            if isinstance(self.classifier_prototype, list):
                clf = deepcopy(self.classifier_prototype[label])
            else:
                clf = deepcopy(self.classifier_prototype)
            clf.fit(x_train, y)

            y_avg = np.average(y)
            if (not self.dynamic_thresholds) or y_avg <= 0.001 or y_avg >= 0.999:
                return {"classifier": clf,
                        "threshold": self.min_threshold[label]}

            proba = clf.predict_proba(x_train)
            assert isinstance(proba, np.ndarray), \
                "classifier should return numpy array"
            assert proba.shape == (x_train.shape[0], 2), \
                "classifier should return (%d,%d)-shaped array, not %s" % (
                    x_train.shape[0], 2, str(proba.shape))

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
            self.thresholds[category, label] = thresholds[k]

            return {"classifier": clf,
                    "threshold": thresholds[k]}

        items = [(category, label)
                 for category in features.keys()
                 for label in [0, 1]]

        if n_jobs == 1:
            results = list(map(_fit, tqdm(items, desc="Fit (primal)")))
        else:
            results = p_map(_fit, items, num_cpus=n_jobs)

        for (idx, (category, label)) in enumerate(items):
            self.thresholds[category, label] = results[idx]["threshold"]
            self.classifiers[category, label] = results[idx]["classifier"]

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
