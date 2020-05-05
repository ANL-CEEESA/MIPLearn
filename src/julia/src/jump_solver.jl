#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MathOptInterface
const MOI = MathOptInterface

function varname_split(varname::String)
    m = match(r"([^[]*)\[(.*)\]", varname)
    return m.captures[1], m.captures[2]
end

@pydef mutable struct JuMPSolver <: InternalSolver
    function __init__(self; optimizer=CPLEX.Optimizer)
        self.optimizer = optimizer
    end

    function add_constraint(self, constraint)
        @error "JuMPSolver: add_constraint not implemented"
    end

    function set_warm_start(self, solution)
        for (basename, subsolution) in solution
            for (idx, value) in subsolution
                value != nothing || continue
                var = self.basename_idx_to_var[basename, idx]
                JuMP.set_start_value(var, value)
            end
        end
    end

    function clear_warm_start(self)
        @error "JuMPSolver: clear_warm_start not implemented"
    end

    function fix(self, solution)
        for (basename, subsolution) in solution
            for (idx, value) in subsolution
                value != nothing || continue
                var = self.basename_idx_to_var[basename, idx]
                JuMP.fix(var, value, force=true)
            end
        end
    end

    function set_instance(self, instance, model)
        self.instance = instance
        self.model = model
        self.var_to_basename_idx = Dict(var => varname_split(JuMP.name(var))
                                        for var in JuMP.all_variables(self.model))
        self.basename_idx_to_var = Dict(varname_split(JuMP.name(var)) => var
                                        for var in JuMP.all_variables(self.model))
        self.bin_vars = [var
                         for var in JuMP.all_variables(self.model)
                         if JuMP.is_binary(var)]
        JuMP.set_optimizer(self.model, self.optimizer)
    end

    function solve(self; tee=false)
        JuMP.optimize!(self.model)
        self._update_solution()

        primal_bound = JuMP.objective_value(self.model)
        dual_bound = JuMP.objective_bound(self.model)

        if JuMP.objective_sense(self.model) == MOI.MIN_SENSE
            sense = "min"
            lower_bound = dual_bound
            upper_bound = primal_bound
        else
            sense = "max"
            lower_bound = primal_bound
            upper_bound = dual_bound
        end

        @show primal_bound, dual_bound

        return Dict("Lower bound" => lower_bound,
                    "Upper bound" => upper_bound,
                    "Sense" => sense,
                    "Wallclock time" => JuMP.solve_time(self.model),
                    "Nodes" => 1,
                    "Log" => nothing,
                    "Warm start value" => nothing)
    end

    function solve_lp(self; tee=false)
        for var in self.bin_vars
            JuMP.unset_binary(var)
            JuMP.set_upper_bound(var, 1.0)
            JuMP.set_lower_bound(var, 0.0)
        end

        JuMP.optimize!(self.model)
        obj_value = JuMP.objective_value(self.model)
        self._update_solution()

        for var in self.bin_vars
            JuMP.set_binary(var)
        end

        return Dict("Optimal value" => obj_value)
    end

    function get_solution(self)
        return self.solution
    end

    function _update_solution(self)
        solution = Dict()
        for var in JuMP.all_variables(self.model)
            basename, idx = self.var_to_basename_idx[var]
            if !haskey(solution, basename)
                solution[basename] = Dict()
            end
            solution[basename][idx] = JuMP.value(var)
        end
        self.solution = solution
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
        JuMP.set_time_limit_sec(self.model, time_limit)
    end
end
