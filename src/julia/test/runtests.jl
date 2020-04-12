#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn

@testset "MIPLearn" begin
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
end