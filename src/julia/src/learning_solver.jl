#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

struct LearningSolver
    py::PyCall.PyObject
end

function LearningSolver(;
                        optimizer,
                        kwargs...,
                       )::LearningSolver
    py = @pycall miplearn.LearningSolver(;
                                         kwargs...,
                                         solver=JuMPSolver(optimizer=optimizer))
    return LearningSolver(py)
end

solve!(solver::LearningSolver, instance; kwargs...) =
    @pycall solver.py.solve(instance; kwargs...)

fit!(solver::LearningSolver, instances; kwargs...) =
    @pycall solver.py.fit(instances; kwargs...)

add!(solver::LearningSolver, component; kwargs...) =
    @pycall solver.py.add(component; kwargs...)

export LearningSolver