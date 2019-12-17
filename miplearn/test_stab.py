# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from .solvers import LearningSolver
from .core import Parameters
import numpy as np
import pyomo.environ as pe
import networkx as nx


class MaxStableSetGenerator:
    """Class that generates random instances of the Maximum Stable Set (MSS) Problem."""
    
    def __init__(self, n_vertices, density=0.1, seed=42):
        self.graph = nx.generators.random_graphs.binomial_graph(n_vertices, density, seed)
        self.base_weights = np.random.rand(self.graph.number_of_nodes()) * 10
        
    def generate(self):
        perturbation = np.random.rand(self.graph.number_of_nodes()) * 0.1
        weights = self.base_weights + perturbation
        return MaxStableSetParameters(self.graph, weights)
    

class MaxStableSetParameters(Parameters):
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
    
    def to_array(self):
        return self.weights


def test_stab():
    generator = MaxStableSetGenerator(n_vertices=100)
    for k in range(5):
        params = generator.generate()
        solver = LearningSolver()
        solver.solve(params)