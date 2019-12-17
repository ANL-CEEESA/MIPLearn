# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from abc import ABC, abstractmethod

class Parameters(ABC):
    """
    Abstract class for holding the data that distinguishes one relevant instance of the problem
    from another.
    
    In the knapsack problem, for example, this class could hold the number of items, their weights
    and costs, as well as the size of the knapsack. Objects implementing this class are able to
    convert themselves into concrete optimization model, which can be solved by a MIPSolver, or
    into 1-dimensional numpy arrays, which can be given to a machine learning model.
    """
    
    @abstractmethod
    def to_model(self):
        """
        Convert the parameters into a concrete optimization model.
        """
        pass
    
    @abstractmethod
    def to_array(self):
        """
        Convert the parameters into a 1-dimensional array.
        
        The array is used by the LearningEnhancedSolver to determine how similar two instances are.
        After some normalization or embedding, it may also be used as input to the machine learning
        models. It must be numerical.
        
        There is not necessarily a one-to-one correspondence between parameters and arrays. The
        array may encode only part of the data necessary to generate a concrete optimization model.
        The entries may also be reductions on the original data. For example, in the knapsack
        problem, an implementation may decide to encode only the average weights, the average prices
        and the size of the knapsack. This technique may be used to guarantee that arrays
        correponding to instances of different sizes have the same dimension.
        """
        pass
