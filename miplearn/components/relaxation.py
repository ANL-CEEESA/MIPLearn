#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging

from miplearn import Component
from miplearn.classifiers.counting import CountingClassifier
from miplearn.components.composite import CompositeComponent
from miplearn.components.steps.convert_tight import ConvertTightIneqsIntoEqsStep
from miplearn.components.steps.drop_redundant import DropRedundantInequalitiesStep
from miplearn.components.steps.relax_integrality import RelaxIntegralityStep

logger = logging.getLogger(__name__)


class RelaxationComponent(Component):
    """
    A Component that tries to build a relaxation that is simultaneously strong and easy
    to solve. Currently, this component is composed by three steps:

    - RelaxIntegralityStep
    - DropRedundantInequalitiesStep
    - ConvertTightIneqsIntoEqsStep

    Parameters
    ----------
    redundant_classifier : Classifier, optional
        Classifier used to predict if a constraint is likely redundant. One deep
        copy of this classifier is made for each constraint category.
    redundant_threshold : float, optional
        If the probability that a constraint is redundant exceeds this threshold, the
        constraint is dropped from the linear relaxation.
    tight_classifier : Classifier, optional
        Classifier used to predict if a constraint is likely to be tight. One deep
        copy of this classifier is made for each constraint category.
    tight_threshold : float, optional
        If the probability that a constraint is tight exceeds this threshold, the
        constraint is converted into an equality constraint.
    slack_tolerance : float, optional
        If a constraint has slack greater than this threshold, then the constraint is
        considered loose. By default, this threshold equals a small positive number to
        compensate for numerical issues.
    check_feasibility : bool, optional
        If true, after the problem is solved, the component verifies that all dropped
        constraints are still satisfied, re-adds the violated ones and resolves the
        problem. This loop continues until either no violations are found, or a maximum
        number of iterations is reached.
    violation_tolerance : float, optional
        If `check_dropped` is true, a constraint is considered satisfied during the
        check if its violation is smaller than this tolerance.
    max_check_iterations : int
        If `check_dropped` is true, set the maximum number of iterations in the lazy
        constraint loop.
    """

    def __init__(
        self,
        redundant_classifier=CountingClassifier(),
        redundant_threshold=0.95,
        tight_classifier=CountingClassifier(),
        tight_threshold=0.95,
        slack_tolerance=1e-5,
        check_feasibility=False,
        violation_tolerance=1e-5,
        max_check_iterations=3,
    ):
        self.steps = [
            RelaxIntegralityStep(),
            DropRedundantInequalitiesStep(
                classifier=redundant_classifier,
                threshold=redundant_threshold,
                slack_tolerance=slack_tolerance,
                violation_tolerance=violation_tolerance,
                max_iterations=max_check_iterations,
                check_feasibility=check_feasibility,
            ),
            ConvertTightIneqsIntoEqsStep(
                classifier=tight_classifier,
                threshold=tight_threshold,
                slack_tolerance=slack_tolerance,
            ),
        ]
        self.composite = CompositeComponent(self.steps)

    def before_solve(self, solver, instance, model):
        self.composite.before_solve(solver, instance, model)

    def after_solve(self, solver, instance, model, stats, training_data):
        self.composite.after_solve(solver, instance, model, stats, training_data)

    def fit(self, training_instances):
        self.composite.fit(training_instances)

    def iteration_cb(self, solver, instance, model):
        return self.composite.iteration_cb(solver, instance, model)
