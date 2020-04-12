#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using PyCall

logging = pyimport("logging")
logging.basicConfig(format="%(levelname)10s %(message)s", level=logging.DEBUG)

@testset "MIPLearn" begin
    include("jump_solver_test.jl")
    include("learning_solver_test.jl")
end