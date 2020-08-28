#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import MIPLearn: get_instance_features,
                 get_variable_features
                 find_violated_lazy_constraints
using JuMP

struct KnapsackData
    weights
    prices
    capacity
end

function to_model(data::KnapsackData)
    model = Model()
    n = length(data.weights)
    @variable(model, x[1:n], Bin)
    @objective(model, Max, sum(x[i] * data.prices[i] for i in 1:n))
    @constraint(model, sum(x[i] * data.weights[i] for i in 1:n) <= data.capacity)
    return model
end

function get_instance_features(data::KnapsackData)
    return [0.]
end


function get_variable_features(data::KnapsackData, var, index)
    return [0.]
end

KnapsackInstance = @Instance(KnapsackData)
