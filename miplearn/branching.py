# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from . import Component
from .transformers import PerVariableTransformer
from abc import ABC, abstractmethod
import numpy as np


class BranchPriorityComponent(Component):
    def __init__(self,
                 initial_priority=None,
                 collect_training_data=True):
        self.priority = initial_priority
        self.transformer = PerVariableTransformer()
        self.collect_training_data = collect_training_data
    
    def before_solve(self, solver, instance, model):
        assert solver.is_persistent, "BranchPriorityComponent requires a persistent solver"
        var_split = self.transformer.split_variables(instance, model)
        for category in var_split.keys():
            var_index_pairs = var_split[category]
            if self.priority is not None:
                from gurobipy import GRB
                for (i, (var, index)) in enumerate(var_index_pairs):
                    gvar = solver.internal_solver._pyomo_var_to_solver_var_map[var[index]]
                    gvar.setAttr(GRB.Attr.BranchPriority, int(self.priority[index]))        

                    
    def after_solve(self, solver, instance, model):
        if self.collect_training_data:
            import subprocess, tempfile, os
            src_dirname = os.path.dirname(os.path.realpath(__file__))
            model_file = tempfile.NamedTemporaryFile(suffix=".lp")
            priority_file = tempfile.NamedTemporaryFile()
            solver.internal_solver.write(model_file.name)
            subprocess.run(["julia",
                            "%s/scripts/branchpriority.jl" % src_dirname,
                            model_file.name,
                            priority_file.name],
                           check=True,
                           capture_output=True)
            self._merge(np.genfromtxt(priority_file.name,
                                      delimiter=',',
                                      dtype=np.float64))
        
    
    def fit(self, solver):
        pass

    
    def merge(self, other_components):
        for comp in other_components:
            if comp.priority is not None:
                self._merge(comp.priority)
            
            
    def _merge(self, priority):
        assert isinstance(priority, np.ndarray)
        if self.priority is None:
            self.priority = priority
        else:
            assert self.priority.shape == priority.shape
            self.priority += priority