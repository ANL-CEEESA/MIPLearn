#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using CPLEX
using MathOptInterface
const MOI = MathOptInterface
using TimerOutputs


mutable struct JuMPSolverData
    basename_idx_to_var
    var_to_basename_idx
    optimizer
    instance
    model
    bin_vars
    solution::Union{Nothing,Dict{String,Dict{String,Float64}}}
end


function varname_split(varname::String)
    m = match(r"([^[]*)\[(.*)\]", varname)
    if m == nothing
        return varname, ""
    end
    return m.captures[1], m.captures[2]
end


"""
    optimize_and_capture_output!(model; tee=tee)

Optimizes a given JuMP model while capturing the solver log, then returns that log.
If tee=true, prints the solver log to the standard output as the optimization takes place.
"""
function optimize_and_capture_output!(model; tee::Bool=false)
    original_stdout = stdout
    rd, wr = redirect_stdout()
    task = @async begin
        log = ""
        while true
            line = String(readavailable(rd))
            isopen(rd) || break
            log *= String(line)
            if tee
                print(original_stdout, line)
                flush(original_stdout)
            end
        end
        return log
    end
    JuMP.unset_silent(model)
    JuMP.optimize!(model)
    sleep(1)
    redirect_stdout(original_stdout)
    close(rd)
    return fetch(task)
end


function solve(data::JuMPSolverData; tee::Bool=false)
    instance, model = data.instance, data.model
    wallclock_time = 0
    found_lazy = []
    log = ""
    while true
        log *= optimize_and_capture_output!(model, tee=tee)
        wallclock_time += JuMP.solve_time(model)
        violations = instance.find_violated_lazy_constraints(model)
        if length(violations) == 0
            break
        end
        append!(found_lazy, violations)
        for v in violations
            instance.build_lazy_constraint(data.model, v)
        end
    end
    update_solution!(data)
    instance.found_violated_lazy_constraints = found_lazy
    instance.found_violated_user_cuts = []
    primal_bound = JuMP.objective_value(model)
    dual_bound = JuMP.objective_bound(model)
    if JuMP.objective_sense(model) == MOI.MIN_SENSE
        sense = "min"
        lower_bound = dual_bound
        upper_bound = primal_bound
    else
        sense = "max"
        lower_bound = primal_bound
        upper_bound = dual_bound
    end
    return Dict("Lower bound" => lower_bound,
                "Upper bound" => upper_bound,
                "Sense" => sense,
                "Wallclock time" => wallclock_time,
                "Nodes" => 1,
                "Log" => log,
                "Warm start value" => nothing)    
end


function solve_lp(data::JuMPSolverData; tee::Bool=false)
    model, bin_vars = data.model, data.bin_vars
    for var in bin_vars
        JuMP.unset_binary(var)
        JuMP.set_upper_bound(var, 1.0)
        JuMP.set_lower_bound(var, 0.0)
    end
    log = optimize_and_capture_output!(model, tee=tee)
    update_solution!(data)
    obj_value = JuMP.objective_value(model)
    for var in bin_vars
        JuMP.set_binary(var)
    end
    return Dict("Optimal value" => obj_value,
                "Log" => log)
end


function update_solution!(data::JuMPSolverData)
    var_to_basename_idx, model = data.var_to_basename_idx, data.model
    solution = Dict{String,Dict{String,Float64}}()
    for var in JuMP.all_variables(model)
        var in keys(var_to_basename_idx) || continue
        basename, idx = var_to_basename_idx[var]
        if !haskey(solution, basename)
            solution[basename] = Dict{String,Float64}()
        end
        solution[basename][idx] = JuMP.value(var)
    end
    data.solution = solution
end


function set_instance!(data::JuMPSolverData, instance, model)
    data.instance = instance
    data.model = model
    data.var_to_basename_idx = Dict(var => varname_split(JuMP.name(var))
                               for var in JuMP.all_variables(model))
    data.basename_idx_to_var = Dict(varname_split(JuMP.name(var)) => var
                               for var in JuMP.all_variables(model))
    data.bin_vars = [var
                     for var in JuMP.all_variables(model)
                     if JuMP.is_binary(var)]
    if data.optimizer != nothing
        JuMP.set_optimizer(model, data.optimizer)
    end
end    


function fix!(data::JuMPSolverData, solution)
    count = 0
    for (basename, subsolution) in solution
        for (idx, value) in subsolution
            value != nothing || continue
            var = data.basename_idx_to_var[basename, idx]
            JuMP.fix(var, value, force=true)
            count += 1
        end
    end
    @info "Fixing $count variables"
end


function set_warm_start!(data::JuMPSolverData, solution)
    count = 0
    for (basename, subsolution) in solution
        for (idx, value) in subsolution
            value != nothing || continue
            var = data.basename_idx_to_var[basename, idx]
            JuMP.set_start_value(var, value)
            count += 1
        end
    end
    @info "Setting warm start values for $count variables"
end    


@pydef mutable struct JuMPSolver <: InternalSolver
    function __init__(self; optimizer)
        self.data = JuMPSolverData(nothing,  # basename_idx_to_var
                                   nothing,  # var_to_basename_idx
                                   optimizer,
                                   nothing,  # instance
                                   nothing,  # model
                                   nothing,  # bin_vars
                                   nothing,  # solution
                                  ) 
    end

    set_warm_start(self, solution) =
        set_warm_start!(self.data, solution)

    fix(self, solution) =
        fix!(self.data, solution)
    
    set_instance(self, instance, model) =
        set_instance!(self.data, instance, model)
    
    solve(self; tee=false) =
        solve(self.data, tee=tee)
    
    solve_lp(self; tee=false) =
        solve_lp(self.data, tee=tee)
    
    get_solution(self) =
        self.data.solution
    
    set_time_limit(self, time_limit) =
        JuMP.set_time_limit_sec(self.data.model, time_limit)

    set_gap_tolerance(self, gap_tolerance) =
        @warn "JuMPSolver: set_gap_tolerance not implemented"
    
    set_node_limit(self) =
        @warn "JuMPSolver: set_node_limit not implemented"
    
    set_threads(self, threads) =
        @warn "JuMPSolver: set_threads not implemented"
    
    set_branching_priorities(self, priorities) =
        @warn "JuMPSolver: set_branching_priorities not implemented"
    
    add_constraint(self, constraint) =
        error("JuMPSolver.add_constraint should never be called")

    clear_warm_start(self) =
        error("JuMPSolver.clear_warm_start should never be called")

end
