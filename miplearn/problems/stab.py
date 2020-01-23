# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import numpy as np
import pyomo.environ as pe
import networkx as nx
from miplearn import Instance
import random


class MaxStableSetGenerator:
    def __init__(self, graph, base_weights, perturbation_scale=1.0):
        self.graph = graph
        self.base_weights = base_weights
        self.perturbation_scale = perturbation_scale
        
    def generate(self):
        perturbation = np.random.rand(self.graph.number_of_nodes()) * self.perturbation_scale
        weights = self.base_weights + perturbation
        return MaxStableSetInstance(self.graph, weights)


class MaxStableSetInstance(Instance):
    def __init__(self, graph, weights):
        self.graph = graph
        self.weights = weights
        self.model = None
        
    def to_model(self):
        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges)
        self.model = model = pe.ConcreteModel()
        model.x = pe.Var(nodes, domain=pe.Binary)
        model.OBJ = pe.Objective(rule=lambda m : sum(m.x[v] * self.weights[v] for v in nodes),
                                 sense=pe.maximize)
        model.edge_eqs = pe.ConstraintList()
        for edge in edges:
            model.edge_eqs.add(model.x[edge[0]] + model.x[edge[1]] <= 1)

        return model
    
    def get_instance_features(self):
        return np.array(self.weights)
    
    def get_variable_features(self, var, index):
        return np.ones(0)
    
    def get_variable_category(self, var, index):
        return index
