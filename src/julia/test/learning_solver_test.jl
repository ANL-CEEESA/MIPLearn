#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn
using CPLEX
using Gurobi

@testset "LearningSolver" begin
    for optimizer in [CPLEX.Optimizer, Gurobi.Optimizer]
        instance = KnapsackInstance([23., 26., 20., 18.],
                                    [505., 352., 458., 220.],
                                    67.0)
        model = instance.to_model()
        solver = LearningSolver(solver=JuMPSolver(optimizer=optimizer),
                                mode="heuristic")
        stats = solver.solve(instance, model)
        @test instance.solution["x"]["1"] == 1.0
        @test instance.solution["x"]["2"] == 0.0
        @test instance.solution["x"]["3"] == 1.0
        @test instance.solution["x"]["4"] == 1.0
        @test instance.lower_bound == 1183.0
        @test instance.upper_bound == 1183.0
        @test round(instance.lp_solution["x"]["1"], digits=3) == 1.000
        @test round(instance.lp_solution["x"]["2"], digits=3) == 0.923
        @test round(instance.lp_solution["x"]["3"], digits=3) == 1.000
        @test round(instance.lp_solution["x"]["4"], digits=3) == 0.000
        @test round(instance.lp_value, digits=3) == 1287.923
        solver.fit([instance])
        solver.solve(instance)
    end
end