# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from . import Component
from .transformers import PerVariableTransformer
from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from p_tqdm import p_map


from tqdm.auto import tqdm
from joblib import Parallel, delayed
import multiprocessing

class BranchPriorityComponent(Component):
    def __init__(self,
                 node_limit=1_000,
                ):
        self.transformer = PerVariableTransformer()
        self.pending_instances = []
        self.x_train = {}
        self.y_train = {}
        self.predictors = {}
        self.node_limit = node_limit
    
    def before_solve(self, solver, instance, model):
        assert solver.is_persistent, "BranchPriorityComponent requires a persistent solver"
        from gurobipy import GRB
        var_split = self.transformer.split_variables(instance, model)
        for category in var_split.keys():
            if category not in self.predictors.keys():
                continue
            var_index_pairs = var_split[category]
            for (i, (var, index)) in enumerate(var_index_pairs):
                x = self._build_x(instance, var, index)
                y = self.predictors[category].predict([x])[0][0]
                gvar = solver.internal_solver._pyomo_var_to_solver_var_map[var[index]]
                gvar.setAttr(GRB.Attr.BranchPriority, int(round(y)))

                    
    def after_solve(self, solver, instance, model):
        self.pending_instances += [instance]
    
    def fit(self, solver, n_jobs=1):
        def _process(instance):
            # Create LP file
            import subprocess, tempfile, os, sys
            lp_file = tempfile.NamedTemporaryFile(suffix=".lp")
            priority_file = tempfile.NamedTemporaryFile()
            model = instance.to_model()
            model.write(lp_file.name)
            
            # Run Julia script
            src_dirname = os.path.dirname(os.path.realpath(__file__))
            priority_file = tempfile.NamedTemporaryFile(mode="r")
            subprocess.run(["julia",
                            "%s/scripts/branchpriority.jl" % src_dirname,
                            lp_file.name,
                            priority_file.name,
                            str(self.node_limit),
                           ],
                           check=True,
                          )
            
            # Parse output
            tokens = [line.strip().split(",") for line in priority_file.readlines()]
            lp_varname_to_priority = {t[0]: int(t[1]) for t in tokens}
        
            # Map priorities back to Pyomo variables
            pyomo_var_to_priority = {}
            from pyomo.core import Var
            from pyomo.core.base.label import TextLabeler
            labeler = TextLabeler()
            symbol_map = list(model.solutions.symbol_map.values())[0]
            
            # Build x_train and y_train
            comp = BranchPriorityComponent()
            for var in model.component_objects(Var):
                for index in var:
                    category = instance.get_variable_category(var, index)
                    if category is None:
                        continue
                    lp_varname = symbol_map.getSymbol(var[index], labeler)
                    var_priority = lp_varname_to_priority[lp_varname]
                    x = self._build_x(instance, var, index)
                    y = np.array([var_priority])
                    
                    if category not in comp.x_train.keys():
                        comp.x_train[category] = np.array([x])
                        comp.y_train[category] = np.array([y])
                    else:
                        comp.x_train[category] = np.vstack([comp.x_train[category], x])
                        comp.y_train[category] = np.vstack([comp.y_train[category], y])
                
            return comp
        
        
        subcomponents = Parallel(n_jobs=n_jobs)(
            delayed(_process)(instance)
            for instance in tqdm(self.pending_instances, desc="Branch priority")
        )
        self.merge(subcomponents)
        self.pending_instances.clear()

        # Retrain ML predictors
        for category in self.x_train.keys():
            x_train = self.x_train[category]
            y_train = self.y_train[category]
            self.predictors[category] = KNeighborsRegressor(n_neighbors=1)
            self.predictors[category].fit(x_train, y_train)
            
        
    def _build_x(self, instance, var, index):
        instance_features = instance.get_instance_features()
        var_features = instance.get_variable_features(var, index)
        return np.hstack([instance_features, var_features])
            
    def merge(self, other_components):
        keys = set(self.x_train.keys())
        for comp in other_components:
            self.pending_instances += comp.pending_instances
            keys = keys.union(set(comp.x_train.keys()))
            
        # Merge x_train and y_train
        for key in keys:
            x_train_submatrices = [comp.x_train[key]
                                   for comp in other_components
                                   if key in comp.x_train.keys()]
            y_train_submatrices = [comp.y_train[key]
                                   for comp in other_components
                                   if key in comp.y_train.keys()]
            if key in self.x_train.keys():
                x_train_submatrices += [self.x_train[key]]
                y_train_submatrices += [self.y_train[key]]
            self.x_train[key] = np.vstack(x_train_submatrices)
            self.y_train[key] = np.vstack(y_train_submatrices)
            
        # Merge trained ML predictors
        for comp in other_components:
            for key in comp.predictors.keys():
                if key not in self.predictors.keys():
                    self.predictors[key] = comp.predictors[key]