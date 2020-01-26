# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import numpy as np
import pyomo.environ as pe
import networkx as nx
from miplearn import Instance
import random
from scipy.stats import uniform, randint, bernoulli
from scipy.stats.distributions import rv_frozen


class MaxWeightStableSetChallengeA:
    """
    - Fixed random graph (200 vertices, 5% density)
    - Uniformly random weights in the [100., 125.] interval
    - 500 training instances
    - 100 test instances
    """
    
    def __init__(self):
        self.generator = MaxWeightStableSetGenerator(w=uniform(loc=100., scale=25.),
                                                     n=randint(low=200, high=201),
                                                     density=uniform(loc=0.05, scale=0.0),
                                                     fix_graph=True)
    
    def get_training_instances(self):
        return self.generator.generate(500)
    
    def get_test_instances(self):
        return self.generator.generate(100)


class MaxWeightStableSetGenerator:
    """Random instance generator for the Maximum-Weight Stable Set Problem.
    
    The generator has two modes of operation. When `fix_graph` is True, the random graph is
    generated only once, during the constructor. Each instance is constructed by generating
    random weights and by randomly deleting vertices and edges of this graph. When `fix_graph`
    is False, a new random graph is created each time an instance is constructed.
    """
    
    def __init__(self,
                 w=uniform(loc=10.0, scale=1.0),
                 pe=bernoulli(1.),
                 pv=bernoulli(1.),
                 n=randint(low=250, high=251),
                 density=uniform(loc=0.05, scale=0.05),
                 fix_graph=True):
        """Initializes the problem generator.
        
        Parameters
        ----------
        w: rv_continuous
            Probability distribution for the vertex weights.
        pe: rv_continuous
            Probability of an edge being deleted. Only used when fix_graph=True.
        pv: rv_continuous
            Probability of a vertex being deleted. Only used when fix_graph=True.
        n: rv_discrete
            Probability distribution for the number of vertices in the random graph.
        density: rv_continuous
            Probability distribution for the density of the random graph.
        """
        assert isinstance(w, rv_frozen), "w should be a SciPy probability distribution"
        assert isinstance(pe, rv_frozen), "pe should be a SciPy probability distribution"
        assert isinstance(pv, rv_frozen), "pv should be a SciPy probability distribution"
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(density, rv_frozen), "density should be a SciPy probability distribution"
        self.w = w
        self.n = n
        self.density = density
        self.fix_graph = fix_graph
        self.graph = None
        if fix_graph:
            self.graph = self._generate_graph()
    
    def generate(self, n_samples):
        def _sample():
            if self.graph is not None:
                graph = self.graph
            else:
                graph = self._generate_graph()
            weights = self.w.rvs(graph.number_of_nodes())
            return MaxWeightStableSetInstance(graph, weights)
        return [_sample() for _ in range(n_samples)]
    
    def _generate_graph(self):
        return nx.generators.random_graphs.binomial_graph(self.n.rvs(), self.density.rvs())


class MaxWeightStableSetInstance(Instance):
    """An instance of the Maximum-Weight Stable Set Problem.
    
    Given a graph G=(V,E) and a weight w_v for each vertex v, the problem asks for a stable
    set S of G maximizing sum(w_v for v in S). A stable set (also called independent set) is
    a subset of vertices, no two of which are adjacent.
    
    This is one of Karp's 21 NP-complete problems.
    """
    
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
