# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import numpy as np
import pyomo.environ as pe
from miplearn import Instance
import random


class TravelingSalesmanChallengeA:
    """Fixed set of cities, small perturbation to travel speed."""
    def __init__():
        self.generator = TravelingSalesmanGenerator(speed=uniform(loc=0.9, scale=0.2),
                                                    x=uniform(loc=0.0, loc=1000.0),
                                                    y=uniform(loc=0.0, loc=1000.0),
                                                    pn=0.0,
                                                    n=randint(low=100, high=100),
                                                    fix_cities=True)
        
    def get_training_instances():
        return self.generator.generate(500)
    
    def get_test_instances():
        return self.generator.generate(100)
    
        
class TravelingSalesmanGenerator:
    """Random generator for the Traveling Salesman Problem.

    The generator starts by randomly selecing n points with coordinates (x_i, y_i), where n, x_i 
    and y_i are random variables. The time required to travel from a pair of cities is calculated
    by: (i) computing the euclidean distance between the cities, (ii) sampling a random variable
    speed_i, (iii) dividing the two numbers.

    If fix_cities is True, the cities and travel times will be calculated only once, during the
    constructor. Each time an instance is generated, however, each city will have probability pv
    of being removed from the list. If fix_cities if False, then the cities and travel times will
    be resampled each time an instance is generated. The probability pn is not used in this case.

    All random variables are independent.
    """

    def __init__(self,
                 speed=uniform(loc=0.75, scale=0.5),
                 x=uniform(loc=0.0, loc=1000.0),
                 y=uniform(loc=0.0, loc=1000.0),
                 pn=0.0,
                 n=randint(low=100, high=100),
                 fix_cities=True):
        """Initializes the problem generator.

        Arguments
        ---------
        speed: rv_continuous
            Probability distribution for travel speed.
        x: rv_continuous
            Probability distribution for the x-coordinate of each city.
        y: rv_continuous
            Probability distribution for the y-coordinate of each city.
        pn: float
            Probability of a city being removed from the list. Only used if fix_cities=True.
        n: rv_discrete
            Probability distribution for the number of cities.
        fix_cities: bool
            If true, cities will be resampled for every generated instance. Otherwise, list of
            cities will be computed once, during the constructor.
        """
        pass
    
    def generate(self, n_samples):
        pass
    

class TravelingSalesmanInstance(Instance):
    """An instance ot the Traveling Salesman Problem.
    
    Given a list of cities and the distance between each pair of cities, the problem asks for the
    shortest route starting at the first city, visiting each other city exactly once, then
    returning to the first city. This problem is a generalization of the Hamiltonian path problem,
    one of Karp's 21 NP-complete problems.
    """
    
    def __init__(self, n_cities, distances):
        assert isinstance(distances, np.array)
        assert distances.shape == (n_cities, n_cities)
        self.n_cities = n_cities
        self.distances = distances
        pass
    
