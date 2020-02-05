# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
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
