# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import miplearn
from miplearn import Instance
import numpy as np
import pyomo.environ as pe
from scipy.stats import uniform, randint, bernoulli
from scipy.stats.distributions import rv_frozen


class ChallengeA:
    """
    - 250 variables, 10 constraints, fixed weights
    - w ~ U(100, 900), jitter ~ U(-100, 100)
    - K = 500, u ~ U(0., 1.)
    - alpha = 0.25
    """
    def __init__(self,
                 seed=42,
                 n_training_instances=300,
                 n_test_instances=50):

        np.random.seed(seed)
        self.gen = MultiKnapsackGenerator(n=randint(low=250, high=251),
                                          m=randint(low=10, high=11),
                                          w=uniform(loc=100.0, scale=900.0),
                                          K=uniform(loc=500.0, scale=0.0),
                                          u=uniform(loc=0.0, scale=1.0),
                                          alpha=uniform(loc=0.25, scale=0.0),
                                          fix_w=True,
                                          w_jitter=uniform(loc=-100.0, scale=200.0),
                                         )
        np.random.seed(seed + 1)
        self.training_instances = self.gen.generate(n_training_instances)
        
        np.random.seed(seed + 2)
        self.test_instances = self.gen.generate(n_test_instances)


class MultiKnapsackInstance(Instance):
    """Representation of the Multidimensional 0-1 Knapsack Problem.
    
    Given a set of n items and m knapsacks, the problem is to find a subset of items S maximizing
    sum(prices[i] for i in S). If selected, each item i occupies weights[i,j] units of space in
    each knapsack j. Furthermore, each knapsack j has limited storage space, given by capacities[j].
    
    This implementation assigns a different category for each decision variable, and therefore
    trains one ML model per variable. It is only suitable when training and test instances have
    same size and items don't shuffle around.
    """
    
    def __init__(self,
                 prices,
                 capacities,
                 weights):
        assert isinstance(prices, np.ndarray)
        assert isinstance(capacities, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert len(weights.shape) == 2
        self.m, self.n = weights.shape
        assert prices.shape == (self.n,)
        assert capacities.shape == (self.m,)
        self.prices = prices
        self.capacities = capacities
        self.weights = weights
        
    def to_model(self):
        model = pe.ConcreteModel()
        model.x = pe.Var(range(self.n), domain=pe.Binary)
        model.OBJ = pe.Objective(rule=lambda model: sum(model.x[j] * self.prices[j]
                                                        for j in range(self.n)),
                                 sense=pe.maximize)
        model.eq_capacity = pe.ConstraintList()
        for i in range(self.m):
            model.eq_capacity.add(sum(model.x[j] * self.weights[i,j]
                                      for j in range(self.n)) <= self.capacities[i])
        
        return model
        
    def get_instance_features(self):
        return np.hstack([
            self.prices,
            self.capacities,
            self.weights.ravel(),
        ])

    def get_variable_features(self, var, index):
        return np.array([])

    def get_variable_category(self, var, index):
        return index


class MultiKnapsackGenerator:
    def __init__(self,
                 n=randint(low=100, high=101),
                 m=randint(low=30, high=31),
                 w=randint(low=0, high=1000),
                 K=randint(low=500, high=500),
                 u=uniform(loc=0.0, scale=1.0),
                 alpha=uniform(loc=0.25, scale=0.0),
                 fix_w=False,
                 w_jitter=randint(low=0, high=1),
                 round=True,
                ):
        """Initialize the problem generator.
        
        Instances have a random number of items (or variables) and a random number of knapsacks
        (or constraints), as specified by the provided probability distributions `n` and `m`,
        respectively. The weight of each item `i` on knapsack `j` is sampled independently from
        the provided distribution `w`. The capacity of knapsack `j` is set to:
        
            alpha_j * sum(w[i,j] for i in range(n)),
            
        where `alpha_j`, the tightness ratio, is sampled from the provided probability
        distribution `alpha`. To make the instances more challenging, the costs of the items
        are linearly correlated to their average weights. More specifically, the weight of each
        item `i` is set to:
        
            sum(w[i,j]/m for j in range(m)) + K * u_i,
            
        where `K`, the correlation coefficient, and `u_i`, the correlation multiplier, are sampled
        from the provided probability distributions. Note that `K` is only sample once for the
        entire instance.
        
        If fix_w=True is provided, then w[i,j] are kept the same in all generated instances. This
        also implies that n and m are kept fixed. Although the prices and capacities are derived
        from w[i,j], as long as u and K are not constants, the generated instances will still not
        be completely identical.

        If a probability distribution w_jitter is provided, then item weights will be set to
        w[i,j] + gamma[i,j] where gamma[i,j] is sampled from w_jitter. When combined with 
        fix_w=True, this argument may be used to generate instances where the weight of each item
        is roughly the same, but not exactly identical, across all instances. The prices of the
        items and the capacities of the knapsacks will be calculated as above, but using these
        perturbed weights instead.
        
        By default, all generated prices, weights and capacities are rounded to the nearest integer
        number. If `round=False` is provided, this rounding will be disabled.
        
        Parameters
        ----------
        n: rv_discrete
            Probability distribution for the number of items (or variables)
        m: rv_discrete
            Probability distribution for the number of knapsacks (or constraints)
        w: rv_continuous
            Probability distribution for the item weights
        K: rv_continuous
            Probability distribution for the profit correlation coefficient
        u: rv_continuous
            Probability distribution for the profit multiplier
        alpha: rv_continuous
            Probability distribution for the tightness ratio
        fix_w: boolean
            If true, weights are kept the same (minus the noise from w_jitter) in all instances
        w_jitter: rv_continuous
            Probability distribution for random noise added to the weights
        round: boolean
            If true, all prices, weights and capacities are rounded to the nearest integer
        """
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(m, rv_frozen), "m should be a SciPy probability distribution"
        assert isinstance(w, rv_frozen), "w should be a SciPy probability distribution"
        assert isinstance(K, rv_frozen), "K should be a SciPy probability distribution"
        assert isinstance(u, rv_frozen), "u should be a SciPy probability distribution"
        assert isinstance(alpha, rv_frozen), "alpha should be a SciPy probability distribution"
        assert isinstance(fix_w, bool), "fix_w should be boolean"
        assert isinstance(w_jitter, rv_frozen), \
                "w_jitter should be a SciPy probability distribution"
        
        self.n = n
        self.m = m
        self.w = w
        self.K = K
        self.u = u
        self.alpha = alpha
        self.w_jitter = w_jitter
        self.round = round
        
        if fix_w:
            self.fix_n = self.n.rvs()
            self.fix_m = self.m.rvs()
            self.fix_w = np.array([self.w.rvs(self.fix_n) for _ in range(self.fix_m)])
        else:
            self.fix_n = None
            self.fix_m = None
            self.fix_w = None
    
    def generate(self, n_samples):
        def _sample():
            if self.fix_w is not None:
                n = self.fix_n
                m = self.fix_m
                w = self.fix_w
            else:
                n = self.n.rvs()
                m = self.m.rvs()
                w = np.array([self.w.rvs(n) for _ in range(m)])
            w = w + np.array([self.w_jitter.rvs(n) for _ in range(m)])
            K = self.K.rvs()
            u = self.u.rvs(n)
            alpha = self.alpha.rvs(m)
            p = np.array([w[:,j].sum() / m + K * u[j] for j in range(n)])
            b = np.array([w[i,:].sum() * alpha[i] for i in range(m)])
            if self.round:
                p = p.round()
                b = b.round()
                w = w.round()
            return MultiKnapsackInstance(p, b, w)
        return [_sample() for _ in range(n_samples)]
    
    