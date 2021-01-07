#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.


class Component:
    """
    A Component is an object which adds functionality to a LearningSolver.

    For better code maintainability, LearningSolver simply delegates most of its
    functionality to Components. Each Component is responsible for exactly one ML
    strategy.
    """

    def before_solve(self, solver, instance, model):
        return

    def after_solve(self, solver, instance, model, results):
        return

    def fit(self, training_instances):
        return

    def iteration_cb(self, solver, instance, model):
        return False

    def lazy_cb(self, solver, instance, model):
        return
