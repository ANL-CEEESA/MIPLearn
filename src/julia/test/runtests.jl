#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using MIPLearn
using Test
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

    function clear_warm_start(self)
    end

    function fix(self, solution)
    end

    function set_gap_tolerance(self, gap_tolerance)
    end

    function set_instance(self, instance, model)
        self.instance = instance
        self.model = model
    end

    function set_node_limit(self)
    end

    function set_threads(self, threads)
    end

    function set_time_limit(self, time_limit)
    end

    function set_warm_start(self, solution)
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
        @constraint(model, sum(x[i] * self.weights[i] for i in 1:n) <= instance.capacity)
        return model
    end

    function get_instance_features(self)
        return [0.]
    end

    function get_variable_features(self, var, index)
        @show var
        return [0.]
    end
end

instance = KnapsackInstance([23., 26., 20., 18.],
                            [505., 352., 458., 220.],
                            67.0)
model = instance.to_model()

solver = JuMPSolver()
solver.set_instance(instance, model)
stats = solver.solve()
# assert len(stats["Log"]) > 100
@test stats["Lower bound"] == 1183.0
@test stats["Upper bound"] == 1183.0
@test stats["Sense"] == "max"
# @test isinstance(stats["Wallclock time"], float)
# @test isinstance(stats["Nodes"], int)

solution = solver.get_solution()
@test solution["x[1]"] == 1.0
@test solution["x[2]"] == 0.0
@test solution["x[3]"] == 1.0
@test solution["x[4]"] == 1.0

# stats = solver.solve_lp()
# @test round(stats["Optimal value"], 3) == 1287.923
#
# solution = solver.get_solution()
# @test round(solution["x"][0], 3) == 1.000
# @test round(solution["x"][1], 3) == 0.923
# @test round(solution["x"][2], 3) == 1.000
# @test round(solution["x"][3], 3) == 0.000

