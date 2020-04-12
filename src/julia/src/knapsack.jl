#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using PyCall

@pydef mutable struct KnapsackInstance <: Instance
    function __init__(self, weights, prices, capacity)
        self.weights = weights
        self.prices = prices
        self.capacity = capacity
    end

    function to_model(self)
        model = Model()
        n = length(self.weights)
        @variable(model, x[1:n], Bin)
        @objective(model, Max, sum(x[i] * self.prices[i] for i in 1:n))
        @constraint(model, sum(x[i] * self.weights[i] for i in 1:n) <= self.capacity)
        return model
    end

    function get_instance_features(self)
        return [0.]
    end

    function get_variable_features(self, var, index)
        return [0.]
    end
end

export KnapsackInstance