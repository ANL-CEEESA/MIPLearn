#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn import Component


class CompositeComponent(Component):
    """
    A Component which redirects each method call to one or more subcomponents.

    Useful for breaking down complex components into smaller classes. See
    RelaxationComponent for a concrete example.

    Parameters
    ----------
    children : list[Component]
        Subcomponents that compose this component.
    """

    def __init__(self, children):
        self.children = children

    def before_solve(self, solver, instance, model):
        for child in self.children:
            child.before_solve(solver, instance, model)

    def after_solve(self, solver, instance, model, results):
        for child in self.children:
            child.after_solve(solver, instance, model, results)

    def fit(self, training_instances):
        for child in self.children:
            child.fit(training_instances)

    def lazy_cb(self, solver, instance, model):
        for child in self.children:
            child.lazy_cb(solver, instance, model)

    def iteration_cb(self, solver, instance, model):
        should_repeat = False
        for child in self.children:
            if child.iteration_cb(solver, instance, model):
                should_repeat = True
        return should_repeat
