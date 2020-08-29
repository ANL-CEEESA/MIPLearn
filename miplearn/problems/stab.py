#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import numpy as np
import pyomo.environ as pe
import networkx as nx
from miplearn import Instance
import random
from scipy.stats import uniform, randint, bernoulli
from scipy.stats.distributions import rv_frozen


class ChallengeA:
    def __init__(self,
                 seed=42,
                 n_training_instances=500,
                 n_test_instances=50,
                ):
        
        np.random.seed(seed)
        self.generator = MaxWeightStableSetGenerator(w=uniform(loc=100., scale=50.),
                                                     n=randint(low=200, high=201),
                                                     p=uniform(loc=0.05, scale=0.0),
                                                     fix_graph=True)
        
        np.random.seed(seed + 1)
        self.training_instances = self.generator.generate(n_training_instances)
        
        np.random.seed(seed + 2)
        self.test_instances = self.generator.generate(n_test_instances)


class MaxWeightStableSetGenerator:
    """Random instance generator for the Maximum-Weight Stable Set Problem.
    
    The generator has two modes of operation. When `fix_graph=True` is provided, one random
    Erdős-Rényi graph $G_{n,p}$ is generated in the constructor, where $n$ and $p$ are sampled
    from user-provided probability distributions `n` and `p`. To generate each instance, the
    generator independently samples each $w_v$ from the user-provided probability distribution `w`.
    
    When `fix_graph=False`, a new random graph is generated for each instance; the remaining
    parameters are sampled in the same way.
    """
    
    def __init__(self,
                 w=uniform(loc=10.0, scale=1.0),
                 n=randint(low=250, high=251),
                 p=uniform(loc=0.05, scale=0.0),
                 fix_graph=True):
        """Initialize the problem generator.
        
        Parameters
        ----------
        w: rv_continuous
            Probability distribution for vertex weights.
        n: rv_discrete
            Probability distribution for parameter $n$ in Erdős-Rényi model.
        p: rv_continuous
            Probability distribution for parameter $p$ in Erdős-Rényi model.
        """
        assert isinstance(w, rv_frozen), "w should be a SciPy probability distribution"
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(p, rv_frozen), "p should be a SciPy probability distribution"
        self.w = w
        self.n = n
        self.p = p
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
        return nx.generators.random_graphs.binomial_graph(self.n.rvs(), self.p.rvs())


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

    def to_model(self):
        nodes = list(self.graph.nodes)
        model = pe.ConcreteModel()
        model.x = pe.Var(nodes, domain=pe.Binary)
        model.OBJ = pe.Objective(expr=sum(model.x[v] * self.weights[v] for v in nodes),
                                 sense=pe.maximize)
        model.clique_eqs = pe.ConstraintList()
        for clique in nx.find_cliques(self.graph):
            model.clique_eqs.add(sum(model.x[i] for i in clique) <= 1)
        return model
    
    def get_instance_features(self):
        return np.ones(0)

    def get_variable_features(self, var, index):
        neighbor_weights = [0] * 15
        neighbor_degrees = [100] * 15
        for n in self.graph.neighbors(index):
            neighbor_weights += [self.weights[n] / self.weights[index]]
            neighbor_degrees += [self.graph.degree(n) / self.graph.degree(index)]
        neighbor_weights.sort(reverse=True)
        neighbor_degrees.sort()
        features = []
        features += neighbor_weights[:5]
        features += neighbor_degrees[:5]
        features += [self.graph.degree(index)]
        return np.array(features)

    def get_variable_category(self, var, index):
        return "default"
