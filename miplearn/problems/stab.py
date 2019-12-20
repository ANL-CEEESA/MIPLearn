# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import numpy as np
import pyomo.environ as pe
import networkx as nx
from miplearn import Instance
import random

class MaxStableSetGenerator:
    def __init__(self, sizes=[50], densities=[0.1]):
        self.sizes = sizes
        self.densities = densities
        
    def generate(self):
        size = random.choice(self.sizes)
        density = random.choice(self.densities)
        self.graph = nx.generators.random_graphs.binomial_graph(size, density)
        weights = np.ones(self.graph.number_of_nodes())
        return MaxStableSetInstance(self.graph, weights)
    

class MaxStableSetInstance(Instance):
    def __init__(self, graph, weights):
        self.graph = graph
        self.weights = weights
        
    def to_model(self):
        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges)
        model = m = pe.ConcreteModel()
        m.x = pe.Var(nodes, domain=pe.Binary)
        m.OBJ = pe.Objective(rule=lambda m : sum(m.x[v] * self.weights[v] for v in nodes),
                              sense=pe.maximize)
        m.edge_eqs = pe.ConstraintList()
        for edge in edges:
            m.edge_eqs.add(m.x[edge[0]] + m.x[edge[1]] <= 1)
        return m
    
    def get_instance_features(self):
        return np.array([
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        ])
    
    def get_variable_features(self, var, index):
        first_neighbors = list(self.graph.neighbors(index))
        second_neighbors = [list(self.graph.neighbors(u)) for u in first_neighbors]
        degree = len(first_neighbors)
        neighbor_degrees = sorted([len(nn) for nn in second_neighbors])
        neighbor_degrees = neighbor_degrees + [100.] * 10
        return np.array([
            degree,
            neighbor_degrees[0] - degree,
            neighbor_degrees[1] - degree,
            neighbor_degrees[2] - degree,
        ])


