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

        solver = LearningSolver(solver=JuMPSolver(optimizer=optimizer))
        stats = solver.solve(instance, model)

        @test stats["Lower bound"] == 1183.0
        @test stats["Upper bound"] == 1183.0
        @test stats["Sense"] == "max"
        @test stats["Wallclock time"] > 0

#         solution = solver.get_solution()
#         @test solution["x[1]"] == 1.0
#         @test solution["x[2]"] == 0.0
#         @test solution["x[3]"] == 1.0
#         @test solution["x[4]"] == 1.0
#
#         stats = solver.solve_lp()
#         @test round(stats["Optimal value"], digits=3) == 1287.923
#
#         solution = solver.get_solution()
#         @test round(solution["x[1]"], digits=3) == 1.000
#         @test round(solution["x[2]"], digits=3) == 0.923
#         @test round(solution["x[3]"], digits=3) == 1.000
#         @test round(solution["x[4]"], digits=3) == 0.000
#
#         solver.fix(Dict(
#             "x[1]" => 1.0,
#             "x[2]" => 0.0,
#             "x[3]" => 0.0,
#             "x[4]" => 1.0,
#         ))
#         stats = solver.solve()
#         @test stats["Lower bound"] == 725.0
#         @test stats["Upper bound"] == 725.0
    end
end