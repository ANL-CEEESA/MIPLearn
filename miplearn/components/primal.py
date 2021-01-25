#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Union, Dict, Callable, List, Hashable, Optional, Any, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from miplearn.classifiers import Classifier
from miplearn.classifiers.adaptive import AdaptiveClassifier
from miplearn.classifiers.threshold import MinPrecisionThreshold, Threshold
from miplearn.components import classifier_evaluation_dict
from miplearn.components.component import Component
from miplearn.extractors import InstanceIterator
from miplearn.instance import Instance
from miplearn.types import TrainingSample, VarIndex, Solution, LearningSolveStats

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
        classifier: Callable[[], Classifier] = lambda: AdaptiveClassifier(),
        mode: str = "exact",
        threshold: Callable[[], Threshold] = lambda: MinPrecisionThreshold(
            [0.98, 0.98]
        ),
    ) -> None:
        assert mode in ["exact", "heuristic"]
        self.mode = mode
        self.classifiers: Dict[Hashable, Classifier] = {}
        self.thresholds: Dict[Hashable, Threshold] = {}
        self.threshold_factory = threshold
        self.classifier_factory = classifier
        self.stats: Dict[str, float] = {}
        self._n_free = 0
        self._n_zero = 0
        self._n_one = 0

    def before_solve(self, solver, instance, model):
        if len(self.thresholds) > 0:
            logger.info("Predicting primal solution...")
            solution = self.predict(instance)

            # Collect prediction statistics
            self._n_free = 0
            self._n_zero = 0
            self._n_one = 0
            for (var, var_dict) in solution.items():
                for (idx, value) in var_dict.items():
                    if value is None:
                        self._n_free += 1
                    else:
                        if value < 0.5:
                            self._n_zero += 1
                        else:
                            self._n_one += 1

            # Provide solution to the solver
            if self.mode == "heuristic":
                solver.internal_solver.fix(solution)
            else:
                solver.internal_solver.set_warm_start(solution)

    def after_solve(
        self,
        solver: "LearningSolver",
        instance: Instance,
        model: Any,
        stats: LearningSolveStats,
        training_data: TrainingSample,
    ) -> None:
        stats["Primal: free"] = self._n_free
        stats["Primal: zero"] = self._n_zero
        stats["Primal: one"] = self._n_one

    def x(
        self,
        instances: Union[List[str], List[Instance]],
    ) -> Dict[Hashable, np.ndarray]:
        return self._build_x_y_dict(instances, self._extract_variable_features)

    def y(
        self,
        instances: Union[List[str], List[Instance]],
    ) -> Dict[Hashable, np.ndarray]:
        return self._build_x_y_dict(instances, self._extract_variable_labels)

    def fit(
        self,
        training_instances: Union[List[str], List[Instance]],
        n_jobs: int = 1,
    ) -> None:
        x = self.x(training_instances)
        y = self.y(training_instances)
        for category in x.keys():
            clf = self.classifier_factory()
            thr = self.threshold_factory()
            clf.fit(x[category], y[category])
            thr.fit(clf, x[category], y[category])
            self.classifiers[category] = clf
            self.thresholds[category] = thr

    def predict(self, instance: Instance) -> Solution:
        assert len(instance.training_data) > 0
        sample = instance.training_data[-1]
        assert "LP solution" in sample
        lp_solution = sample["LP solution"]
        assert lp_solution is not None

        # Initialize empty solution
        solution: Solution = {}
        for (var_name, var_dict) in lp_solution.items():
            solution[var_name] = {}
            for (idx, lp_value) in var_dict.items():
                solution[var_name][idx] = None

        # Compute y_pred
        x = self.x([instance])
        y_pred = {}
        for category in x.keys():
            assert category in self.classifiers, (
                f"Classifier for category {category} has not been trained. "
                f"Please call component.fit before component.predict."
            )
            proba = self.classifiers[category].predict_proba(x[category])
            thr = self.thresholds[category].predict(x[category])
            y_pred[category] = np.vstack(
                [
                    proba[:, 0] > thr[0],
                    proba[:, 1] > thr[1],
                ]
            ).T

        # Convert y_pred into solution
        category_offset: Dict[Hashable, int] = {cat: 0 for cat in x.keys()}
        for (var_name, var_dict) in lp_solution.items():
            for (idx, lp_value) in var_dict.items():
                category = instance.get_variable_category(var_name, idx)
                offset = category_offset[category]
                category_offset[category] += 1
                if y_pred[category][offset, 0]:
                    solution[var_name][idx] = 0.0
                if y_pred[category][offset, 1]:
                    solution[var_name][idx] = 1.0

        return solution

    def evaluate(self, instances):
        ev = {"Fix zero": {}, "Fix one": {}}
        for instance_idx in tqdm(
            range(len(instances)),
            desc="Evaluate (primal)",
        ):
            instance = instances[instance_idx]
            solution_actual = instance.training_data[0]["Solution"]
            solution_pred = self.predict(instance)

            vars_all, vars_one, vars_zero = set(), set(), set()
            pred_one_positive, pred_zero_positive = set(), set()
            for (varname, var_dict) in solution_actual.items():
                if varname not in solution_pred.keys():
                    continue
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

            ev["Fix zero"][instance_idx] = classifier_evaluation_dict(
                tp_zero, tn_zero, fp_zero, fn_zero
            )
            ev["Fix one"][instance_idx] = classifier_evaluation_dict(
                tp_one, tn_one, fp_one, fn_one
            )
        return ev

    @staticmethod
    def _build_x_y_dict(
        instances: Union[List[str], List[Instance]],
        extract: Callable[
            [
                Instance,
                TrainingSample,
                str,
                VarIndex,
                Optional[float],
            ],
            Union[List[bool], List[float]],
        ],
    ) -> Dict[Hashable, np.ndarray]:
        result: Dict[Hashable, List] = {}
        for instance in InstanceIterator(instances):
            assert isinstance(instance, Instance)
            for sample in instance.training_data:
                # Skip training samples without solution
                if "LP solution" not in sample:
                    continue
                if sample["LP solution"] is None:
                    continue

                # Iterate over all variables
                for (var, var_dict) in sample["LP solution"].items():
                    for (idx, lp_value) in var_dict.items():
                        category = instance.get_variable_category(var, idx)
                        if category is None:
                            continue
                        if category not in result:
                            result[category] = []
                        result[category] += [
                            extract(
                                instance,
                                sample,
                                var,
                                idx,
                                lp_value,
                            )
                        ]

        # Convert result to numpy arrays and return
        return {c: np.array(ft) for (c, ft) in result.items()}

    @staticmethod
    def _extract_variable_features(
        instance: Instance,
        sample: TrainingSample,
        var: str,
        idx: VarIndex,
        lp_value: Optional[float],
    ) -> Union[List[bool], List[float]]:
        features = instance.get_variable_features(var, idx)
        if lp_value is None:
            return features
        else:
            return features + [lp_value]

    @staticmethod
    def _extract_variable_labels(
        instance: Instance,
        sample: TrainingSample,
        var: str,
        idx: VarIndex,
        lp_value: Optional[float],
    ) -> Union[List[bool], List[float]]:
        assert "Solution" in sample
        solution = sample["Solution"]
        assert solution is not None
        opt_value = solution[var][idx]
        assert opt_value is not None
        assert 0.0 - 1e-5 <= opt_value <= 1.0 + 1e-5, (
            f"Variable {var} has non-binary value {opt_value} in the optimal solution. "
            f"Predicting values of non-binary variables is not currently supported. "
            f"Please set its category to None."
        )
        return [opt_value < 0.5, opt_value > 0.5]
