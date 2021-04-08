#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import (
    Dict,
    List,
    Hashable,
    Any,
    TYPE_CHECKING,
    Tuple,
)

import numpy as np
from overrides import overrides

from miplearn.classifiers import Classifier
from miplearn.classifiers.adaptive import AdaptiveClassifier
from miplearn.classifiers.threshold import MinPrecisionThreshold, Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.features import TrainingSample, Features
from miplearn.instance.base import Instance
from miplearn.types import (
    LearningSolveStats,
    Category,
    Solution,
)

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

    @overrides
    def before_solve_mip(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        features: Features,
        training_data: TrainingSample,
    ) -> None:
        logger.info("Predicting primal solution...")

        # Do nothing if models are not trained
        if len(self.classifiers) == 0:
            logger.info("Classifiers not fitted. Skipping.")
            return

        # Predict solution and provide it to the solver
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
        for (var_name, value) in solution.items():
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
        solution: Solution = {v: None for v in instance.features.variables.keys()}
        category_offset: Dict[Hashable, int] = {cat: 0 for cat in x.keys()}
        for (var_name, var_features) in instance.features.variables.items():
            category = var_features.category
            if category not in category_offset:
                continue
            offset = category_offset[category]
            category_offset[category] += 1
            if y_pred[category][offset, 0]:
                solution[var_name] = 0.0
            if y_pred[category][offset, 1]:
                solution[var_name] = 1.0

        return solution

    @overrides
    def sample_xy(
        self,
        instance: Instance,
        sample: TrainingSample,
    ) -> Tuple[Dict[Category, List[List[float]]], Dict[Category, List[List[float]]]]:
        assert instance.features.variables is not None
        x: Dict = {}
        y: Dict = {}
        for (var_name, var_features) in instance.features.variables.items():
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
                lp_value = sample.lp_solution[var_name]
                if lp_value is not None:
                    f += [lp_value]
            x[category] += [f]
            if sample.solution is not None:
                opt_value = sample.solution[var_name]
                assert opt_value is not None
                assert 0.0 - 1e-5 <= opt_value <= 1.0 + 1e-5, (
                    f"Variable {var_name} has non-binary value {opt_value} in the "
                    "optimal solution. Predicting values of non-binary "
                    "variables is not currently supported. Please set its "
                    "category to None."
                )
                y[category] += [[opt_value < 0.5, opt_value >= 0.5]]
        return x, y

    @overrides
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
        for (var_name, value_actual) in solution_actual.items():
            assert value_actual is not None
            vars_all.add(var_name)
            if value_actual > 0.5:
                vars_one.add(var_name)
            else:
                vars_zero.add(var_name)
            value_pred = solution_pred[var_name]
            if value_pred is not None:
                if value_pred > 0.5:
                    pred_one_positive.add(var_name)
                else:
                    pred_zero_positive.add(var_name)
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

    @overrides
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
