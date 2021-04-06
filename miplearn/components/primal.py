#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import (
    Dict,
    List,
    Hashable,
    Optional,
    Any,
    TYPE_CHECKING,
    Tuple,
)

import numpy as np

from miplearn.classifiers import Classifier
from miplearn.classifiers.adaptive import AdaptiveClassifier
from miplearn.classifiers.threshold import MinPrecisionThreshold, Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.instance import Instance
from miplearn.types import (
    Solution,
    LearningSolveStats,
)
from miplearn.features import TrainingSample, Features

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from miplearn.solvers.learning import LearningSolver


class PrimalSolutionComponent(Component):
    """
    A component that predicts the optimal primal values for the binary decision
    variables.

    In exact mode, predicted primal solutions are provided to the solver as MIP
    starts. In heuristic mode, this component fixes the decision variables to their
    predicted values.
    """

    def __init__(
        self,
        classifier: Classifier = AdaptiveClassifier(),
        mode: str = "exact",
        threshold: Threshold = MinPrecisionThreshold([0.98, 0.98]),
    ) -> None:
        assert isinstance(classifier, Classifier)
        assert isinstance(threshold, Threshold)
        assert mode in ["exact", "heuristic"]
        self.mode = mode
        self.classifiers: Dict[Hashable, Classifier] = {}
        self.thresholds: Dict[Hashable, Threshold] = {}
        self.threshold_prototype = threshold
        self.classifier_prototype = classifier

    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        # Do nothing if models are not trained
        if len(self.classifiers) == 0:
            return

        # Predict solution and provide it to the solver
        logger.info("Predicting MIP solution...")
        solution = self.sample_predict(instance, training_data)
        assert solver.internal_solver is not None
        if self.mode == "heuristic":
            solver.internal_solver.fix(solution)
        else:
            solver.internal_solver.set_warm_start(solution)

        # Update statistics
        stats["Primal: Free"] = 0
        stats["Primal: Zero"] = 0
        stats["Primal: One"] = 0
        for (var, var_dict) in solution.items():
            for (idx, value) in var_dict.items():
                if value is None:
                    stats["Primal: Free"] += 1
                else:
                    if value < 0.5:
                        stats["Primal: Zero"] += 1
                    else:
                        stats["Primal: One"] += 1
        logger.info(
            f"Predicted: free: {stats['Primal: Free']}, "
            f"zero: {stats['Primal: Zero']}, "
            f"one: {stats['Primal: One']}"
        )

    def sample_predict(
        self,
        instance: Instance,
        sample: TrainingSample,
    ) -> Solution:
        assert instance.features.variables is not None

        # Initialize empty solution
        solution: Solution = {}
        for (var_name, var_dict) in instance.features.variables.items():
            solution[var_name] = {}
            for idx in var_dict.keys():
                solution[var_name][idx] = None

        # Compute y_pred
        x, _ = self.sample_xy(instance, sample)
        y_pred = {}
        for category in x.keys():
            assert category in self.classifiers, (
                f"Classifier for category {category} has not been trained. "
                f"Please call component.fit before component.predict."
            )
            xc = np.array(x[category])
            proba = self.classifiers[category].predict_proba(xc)
            thr = self.thresholds[category].predict(xc)
            y_pred[category] = np.vstack(
                [
                    proba[:, 0] >= thr[0],
                    proba[:, 1] >= thr[1],
                ]
            ).T

        # Convert y_pred into solution
        category_offset: Dict[Hashable, int] = {cat: 0 for cat in x.keys()}
        for (var_name, var_dict) in instance.features.variables.items():
            for (idx, var_features) in var_dict.items():
                category = var_features.category
                offset = category_offset[category]
                category_offset[category] += 1
                if y_pred[category][offset, 0]:
                    solution[var_name][idx] = 0.0
                if y_pred[category][offset, 1]:
                    solution[var_name][idx] = 1.0

        return solution

    @staticmethod
    def sample_xy(
        instance: Instance,
        sample: TrainingSample,
    ) -> Tuple[Dict[Hashable, List[List[float]]], Dict[Hashable, List[List[float]]]]:
        assert instance.features.variables is not None
        x: Dict = {}
        y: Dict = {}
        solution: Optional[Solution] = None
        if sample.solution is not None:
            solution = sample.solution
        for (var_name, var_dict) in instance.features.variables.items():
            for (idx, var_features) in var_dict.items():
                category = var_features.category
                if category is None:
                    continue
                if category not in x.keys():
                    x[category] = []
                    y[category] = []
                f: List[float] = []
                assert var_features.user_features is not None
                f += var_features.user_features
                if sample.lp_solution is not None:
                    lp_value = sample.lp_solution[var_name][idx]
                    if lp_value is not None:
                        f += [lp_value]
                x[category] += [f]
                if solution is not None:
                    opt_value = solution[var_name][idx]
                    assert opt_value is not None
                    assert 0.0 - 1e-5 <= opt_value <= 1.0 + 1e-5, (
                        f"Variable {var_name} has non-binary value {opt_value} in the "
                        "optimal solution. Predicting values of non-binary "
                        "variables is not currently supported. Please set its "
                        "category to None."
                    )
                    y[category] += [[opt_value < 0.5, opt_value >= 0.5]]
        return x, y

    def sample_evaluate(
        self,
        instance: Instance,
        sample: TrainingSample,
    ) -> Dict[Hashable, Dict[str, float]]:
        solution_actual = sample.solution
        assert solution_actual is not None
        solution_pred = self.sample_predict(instance, sample)
        vars_all, vars_one, vars_zero = set(), set(), set()
        pred_one_positive, pred_zero_positive = set(), set()
        for (varname, var_dict) in solution_actual.items():
            if varname not in solution_pred.keys():
                continue
            for (idx, value_actual) in var_dict.items():
                assert value_actual is not None
                vars_all.add((varname, idx))
                if value_actual > 0.5:
                    vars_one.add((varname, idx))
                else:
                    vars_zero.add((varname, idx))
                value_pred = solution_pred[varname][idx]
                if value_pred is not None:
                    if value_pred > 0.5:
                        pred_one_positive.add((varname, idx))
                    else:
                        pred_zero_positive.add((varname, idx))
        pred_one_negative = vars_all - pred_one_positive
        pred_zero_negative = vars_all - pred_zero_positive
        return {
            0: classifier_evaluation_dict(
                tp=len(pred_zero_positive & vars_zero),
                tn=len(pred_zero_negative & vars_one),
                fp=len(pred_zero_positive & vars_one),
                fn=len(pred_zero_negative & vars_zero),
            ),
            1: classifier_evaluation_dict(
                tp=len(pred_one_positive & vars_one),
                tn=len(pred_one_negative & vars_zero),
                fp=len(pred_one_positive & vars_zero),
                fn=len(pred_one_negative & vars_one),
            ),
        }

    def fit_xy(
        self,
        x: Dict[Hashable, np.ndarray],
        y: Dict[Hashable, np.ndarray],
    ) -> None:
        for category in x.keys():
            clf = self.classifier_prototype.clone()
            thr = self.threshold_prototype.clone()
            clf.fit(x[category], y[category])
            thr.fit(clf, x[category], y[category])
            self.classifiers[category] = clf
            self.thresholds[category] = thr
