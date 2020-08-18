#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod
import pickle, gzip, json


class Instance(ABC):
    """
    Abstract class holding all the data necessary to generate a concrete model of the problem.
    
    In the knapsack problem, for example, this class could hold the number of items, their weights
    and costs, as well as the size of the knapsack. Objects implementing this class are able to
    convert themselves into a concrete optimization model, which can be optimized by a solver, or
    into arrays of features, which can be provided as inputs to machine learning models.
    """

    @abstractmethod
    def to_model(self):
        """
        Returns a concrete Pyomo model corresponding to this instance.
        """
        pass

    @abstractmethod
    def get_instance_features(self):
        """
        Returns a 1-dimensional Numpy array of (numerical) features describing the entire instance.
        
        The array is used by LearningSolver to determine how similar two instances are. It may also
        be used to predict, in combination with variable-specific features, the values of binary
        decision variables in the problem.
        
        There is not necessarily a one-to-one correspondence between models and instance features:
        the features may encode only part of the data necessary to generate the complete model.
        Features may also be statistics computed from the original data. For example, in the
        knapsack problem, an implementation may decide to provide as instance features only
        the average weights, average prices, number of items and the size of the knapsack.
        
        The returned array MUST have the same length for all relevant instances of the problem. If
        two instances map into arrays of different lengths, they cannot be solved by the same
        LearningSolver object.
        """
        pass

    @abstractmethod
    def get_variable_features(self, var, index):
        """
        Returns a 1-dimensional array of (numerical) features describing a particular decision
        variable.
        
        The argument `var` is a pyomo.core.Var object, which represents a collection of decision
        variables. The argument `index` specifies which variable in the collection is the relevant
        one.
        
        In combination with instance features, variable features are used by LearningSolver to
        predict, among other things, the optimal value of each decision variable before the
        optimization takes place. In the knapsack problem, for example, an implementation could
        provide as variable features the weight and the price of a specific item.
        
        Like instance features, the arrays returned by this method MUST have the same length for
        all variables within the same category, for all relevant instances of the problem.
        """
        pass

    def get_variable_category(self, var, index):
        """
        Returns the category (a string, an integer or any hashable type) for each decision
        variable.
        
        If two variables have the same category, LearningSolver will use the same internal ML
        model to predict the values of both variables. By default, all variables belong to the
        "default" category, and therefore only one ML model is used for all variables.
        
        If the returned category is None, ML models will ignore the variable.
        """
        return "default"

    def find_violated_lazy_constraints(self, model):
        """
        Returns lazy constraint violations found for the current solution.
        
        After solving a model, LearningSolver will ask the instance to identify which lazy
        constraints are violated by the current solution. For each identified violation,
        LearningSolver will then call the build_lazy_constraint, add the generated Pyomo
        constraint to the model, then resolve the problem. The process repeats until no further
        lazy constraint violations are found.
        
        Each "violation" is simply a string, a tuple or any other hashable type which allows the
        instance to identify unambiguously which lazy constraint should be generated. In the
        Traveling Salesman Problem, for example, a subtour violation could be a frozen set
        containing the cities in the subtour.
        
        For a concrete example, see TravelingSalesmanInstance.
        """
        return []

    def build_lazy_constraint(self, model, violation):
        """
        Returns a Pyomo constraint which fixes a given violation.
        
        This method is typically called immediately after find_violated_lazy_constraints. The violation object
        provided to this method is exactly the same object returned earlier by find_violated_lazy_constraints.
        After some training, LearningSolver may decide to proactively build some lazy constraints
        at the beginning of the optimization process, before a solution is even available. In this
        case, build_lazy_constraints will be called without a corresponding call to
        find_violated_lazy_constraints.
        
        The implementation should not directly add the constraint to the model. The constraint
        will be added by LearningSolver after the method returns.
        
        For a concrete example, see TravelingSalesmanInstance.
        """
        pass

    def find_violated_user_cuts(self, model):
        return []

    def build_user_cut(self, model, violation):
        pass
    
    def load(self, filename):
        with gzip.GzipFile(filename, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))
        self.__dict__ = data
        
    def dump(self, filename):
        data = json.dumps(self.__dict__, indent=2).encode('utf-8')
        with gzip.GzipFile(filename, 'w') as f:
            f.write(data)