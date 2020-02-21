# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright © 2020, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.
# Written by Alinson S. Xavier <axavier@anl.gov>

from abc import ABC, abstractmethod


class Component(ABC):
    """
    A Component is an object which adds functionality to a LearningSolver.
    """
    
    @abstractmethod
    def before_solve(self, solver, instance, model):
        pass
    
    @abstractmethod
    def after_solve(self, solver, instance, model):
        pass
    
    @abstractmethod
    def merge(self, other):
        pass
    
    @abstractmethod
    def fit(self, solver):
        pass
