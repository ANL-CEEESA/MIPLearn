#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

__precompile__(false)
module MIPLearn
    using PyCall
    miplearn = pyimport("miplearn")
    Instance = miplearn.Instance
    LearningSolver = miplearn.LearningSolver
    InternalSolver = miplearn.solvers.internal.InternalSolver

    include("jump_solver.jl")
    include("knapsack.jl")

    export Instance, LearningSolver, InternalSolver, JuMPSolver
end # module
