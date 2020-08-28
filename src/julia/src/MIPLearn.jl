#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

__precompile__(false)
module MIPLearn

using PyCall
miplearn = pyimport("miplearn")
Instance = miplearn.Instance
BenchmarkRunner = miplearn.BenchmarkRunner

macro pycall(expr)
    quote
        err_msg = nothing
        result = nothing
        try
            result = $(esc(expr))
        catch err
            args = err.val.args[1]
            if (err isa PyCall.PyError) && (args isa String) && startswith(args, "Julia")
                err_msg = replace(args, r"Stacktrace.*" => "")
            else
                rethrow(err)
            end
        end
        if err_msg != nothing
            error(err_msg)
        end
        result
    end
end

include("log.jl")
include("jump_solver.jl")
include("learning_solver.jl")
include("instance.jl")

export Instance, BenchmarkRunner

end # module
