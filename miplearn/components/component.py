#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod


class Component(ABC):
    """
    A Component is an object which adds functionality to a LearningSolver.
    """
    
    @abstractmethod
    def before_solve(self, solver, instance, model):
        pass
    
    @abstractmethod
    def after_solve(self, solver, instance, model, results):
        pass
    
    @abstractmethod
    def fit(self, training_instances):
        pass

    def after_iteration(self, solver, instance, model):
        return False
