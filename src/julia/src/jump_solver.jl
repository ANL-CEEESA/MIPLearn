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
    solution
end

function varname_split(varname::String)
    m = match(r"([^[]*)\[(.*)\]", varname)
    if m == nothing
        return varname, ""
    end
    return m.captures[1], m.captures[2]
end

@pydef mutable struct JuMPSolver <: InternalSolver
    function __init__(self; optimizer=nothing)
        self.data = JuMPSolverData(nothing,  # basename_idx_to_var
                                   nothing,  # var_to_basename_idx
                                   optimizer,
                                   nothing,  # instance
                                   nothing,  # model
                                   nothing,  # bin_vars
                                   nothing,  # solution
                                  ) 
    end

    function add_constraint(self, constraint)
        @error "JuMPSolver: add_constraint not implemented"
    end

    function set_warm_start(self, solution)
        basename_idx_to_var = self.data.basename_idx_to_var
        for (basename, subsolution) in solution
            for (idx, value) in subsolution
                value != nothing || continue
                var = basename_idx_to_var[basename, idx]
                JuMP.set_start_value(var, value)
            end
        end
    end

    function clear_warm_start(self)
        @error "JuMPSolver: clear_warm_start not implemented"
    end

    function fix(self, solution)
        @timeit "fix" begin
            basename_idx_to_var = self.data.basename_idx_to_var
            for (basename, subsolution) in solution
                for (idx, value) in subsolution
                    value != nothing || continue
                    var = basename_idx_to_var[basename, idx]
                    JuMP.fix(var, value, force=true)
                end
            end
        end
    end

    function set_instance(self, instance, model)
        @timeit "set_instance" begin
            self.data.instance = instance
            self.data.model = model
            self.data.var_to_basename_idx = Dict(var => varname_split(JuMP.name(var))
                                            for var in JuMP.all_variables(model))
            self.data.basename_idx_to_var = Dict(varname_split(JuMP.name(var)) => var
                                            for var in JuMP.all_variables(model))
            self.data.bin_vars = [var
                             for var in JuMP.all_variables(model)
                             if JuMP.is_binary(var)]
            if self.data.optimizer != nothing
                JuMP.set_optimizer(model, self.data.optimizer)
            end
        end
    end

    function solve(self; tee=false)
        @timeit "solve" begin
            instance, model = self.data.instance, self.data.model
            wallclock_time = 0
            found_lazy = []
            while true
                @timeit "optimize!" begin
                    JuMP.optimize!(model)
                end
                wallclock_time += JuMP.solve_time(model)
                @timeit "find_violated_lazy_constraints" begin
                    violations = instance.find_violated_lazy_constraints(model)
                end
                @info "$(length(violations)) violations found"
                if length(violations) == 0
                    break
                end
                append!(found_lazy, violations)
                for v in violations
                    instance.build_lazy_constraint(self.data.model, v)
                end
            end
            @timeit "update solution" begin
                self._update_solution()
                instance.found_violated_lazy_constraints = found_lazy
                instance.found_violated_user_cuts = []
            end
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
        end
        return Dict("Lower bound" => lower_bound,
                    "Upper bound" => upper_bound,
                    "Sense" => sense,
                    "Wallclock time" => wallclock_time,
                    "Nodes" => 1,
                    "Log" => nothing,
                    "Warm start value" => nothing)
    end

    function solve_lp(self; tee=false)
        @timeit "solve_lp" begin
            model = self.data.model
            bin_vars = self.data.bin_vars
            @timeit "unset_binary" begin
                for var in bin_vars
                    JuMP.unset_binary(var)
                    JuMP.set_upper_bound(var, 1.0)
                    JuMP.set_lower_bound(var, 0.0)
                end
            end
            @timeit "optimize" begin
                JuMP.optimize!(model)
            end
            @timeit "update solution" begin
                self._update_solution()
            end
            obj_value = JuMP.objective_value(model)
            @timeit "set_binary" begin
                for var in bin_vars
                    JuMP.set_binary(var)
                end
            end
        end
        return Dict("Optimal value" => obj_value)
    end

    function get_solution(self)
        return self.data.solution
    end

    function _update_solution(self)
        var_to_basename_idx, model = self.data.var_to_basename_idx, self.data.model
        solution = Dict()
        for var in JuMP.all_variables(model)
            var in keys(var_to_basename_idx) || continue
            basename, idx = var_to_basename_idx[var]
            if !haskey(solution, basename)
                solution[basename] = Dict()
            end
            solution[basename][idx] = JuMP.value(var)
        end
        self.data.solution = solution
    end

    function set_gap_tolerance(self, gap_tolerance)
        @error "JuMPSolver: set_gap_tolerance not implemented"
    end

    function set_node_limit(self)
        @error "JuMPSolver: set_node_limit not implemented"
    end

    function set_threads(self, threads)
        @error "JuMPSolver: set_threads not implemented"
    end

    function set_branching_priorities(self, priorities)
        @error "JuMPSolver: set_branching_priorities not implemented"
    end

    function set_time_limit(self, time_limit)
        JuMP.set_time_limit_sec(self.data.model, time_limit)
    end
end
