#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

__precompile__(false)
module MIPLearn

using JuMP
using Gurobi
using PyCall
using MathOptInterface
const MOI = MathOptInterface

miplearn = pyimport("miplearn")
Instance = miplearn.Instance
LearningSolver = miplearn.LearningSolver
InternalSolver = miplearn.solvers.internal.InternalSolver

@pydef mutable struct JuMPSolver <: InternalSolver
    function add_constraint(self, constraint)
    end

    function set_warm_start(self, solution)
    end

    function clear_warm_start(self)
    end

    function fix(self, solution)
    end

    function set_instance(self, instance, model)
        self.instance = instance
        self.model = model
    end

    function solve(self; tee=false)
        JuMP.set_optimizer(self.model, Gurobi.Optimizer)
        JuMP.optimize!(self.model)

        primal_bound = JuMP.objective_value(self.model)
        dual_bound = JuMP.objective_bound(self.model)

        if JuMP.objective_sense(self.model) == MOI.MIN_SENSE
            sense = "min"
            lower_bound = dual_bound
            upper_bound = primal_bound
        else
            sense = "max"
            lower_bound = primal_bound
            upper_bound = dual_bound
        end

        @show primal_bound, dual_bound

        return Dict("Lower bound" => lower_bound,
                    "Upper bound" => upper_bound,
                    "Sense" => sense)
    end

    function solve_lp(self; tee=false)
    end

    function get_solution(self)
        return Dict(JuMP.name(var) => JuMP.value(var)
                    for var in JuMP.all_variables(self.model))
    end

    function set_gap_tolerance(self, gap_tolerance)
    end

    function set_node_limit(self)
    end

    function set_threads(self, threads)
    end

    function set_time_limit(self, time_limit)
    end
end

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

export JuMPSolver, KnapsackInstance

end # module
