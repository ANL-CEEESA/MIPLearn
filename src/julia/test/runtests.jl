#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn

MIPLearn.setup_logger()

@testset "MIPLearn" begin
    include("knapsack.jl")
    include("jump_solver_test.jl")
    include("learning_solver_test.jl")
end